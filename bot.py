# bot.py
import os
import re
import logging
import asyncio
import json
import sys
import urllib.parse # Needed for URL encoding

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)

# Import specific libraries
# from youtube_transcript_api import YouTubeTranscriptApi # No longer primary for YT
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# --- Environment Variable Names (Constants) ---
TELEGRAM_TOKEN_ENV = "TELEGRAM_TOKEN"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
URLTOTEXT_API_KEY_ENV = "URLTOTEXT_API_KEY"
SUPADATA_API_KEY_ENV = "SUPADATA_API_KEY" # <<< NEW KEY NAME

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def is_youtube_url(url):
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)([\w-]{11})'
    return bool(re.search(youtube_regex, url))

# Not strictly needed for Supadata/Gemini URL, but harmless to keep for now
# def extract_youtube_id(url):
#     youtube_id_regex = r'(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})'
#     match = re.search(youtube_id_regex, url)
#     if match:
#         return match.group(1)
#     logger.warning(f"Could not extract YouTube ID from URL: {url}")
#     return None

# --- Content Fetching Functions ---

# <<< NEW FUNCTION for Supadata >>>
async def get_youtube_transcript_via_supadata(youtube_url: str, api_key: str):
    """Attempts to fetch YouTube transcript using the Supadata API."""
    if not youtube_url: logger.error("[Supadata] get_youtube_transcript_via_supadata called with no URL"); return None
    if not api_key:
        logger.error("[Supadata] Supadata API key was not provided for this call.")
        return None

    logger.info(f"[Supadata] Attempting to fetch transcript for: {youtube_url}")
    # Ensure the URL is properly encoded for a query parameter
    encoded_url = urllib.parse.quote(youtube_url, safe='')
    # Construct the API endpoint URL (assuming /youtube/transcript?url=...)
    api_endpoint = f"https://api.supadata.ai/v1/youtube/transcript?url={encoded_url}"

    headers = {
        "x-api-key": api_key, # Correct header name based on Supadata docs
        "Accept": "application/json" # Generally good practice
    }

    try:
        logger.debug(f"[Supadata] Sending request to Supadata API: {api_endpoint}")
        # Run synchronous GET request in a thread
        response = await asyncio.to_thread(requests.get, api_endpoint, headers=headers, timeout=45) # Generous timeout
        logger.debug(f"[Supadata] Received status code {response.status_code} from Supadata API for {youtube_url}")

        if response.status_code == 200:
            try:
                data = response.json()
                # Assuming the transcript is in response['data']['transcript'] - ADJUST IF NEEDED based on actual API response
                transcript = data.get("data", {}).get("transcript")
                if transcript:
                    logger.info(f"[Supadata] Successfully fetched transcript via API for {youtube_url}. Length: {len(transcript)}")
                    return transcript.strip()
                else:
                    logger.warning(f"[Supadata] Supadata API returned success but transcript was empty for {youtube_url}. Response: {data}")
                    return None
            except json.JSONDecodeError:
                logger.error(f"[Supadata] Failed to decode JSON response from Supadata API for {youtube_url}. Response text: {response.text[:500]}...")
                return None
            except Exception as e:
                logger.error(f"[Supadata] Error processing successful Supadata API response for {youtube_url}: {e}", exc_info=True)
                return None
        # Handle specific Supadata error codes (check their docs for exact meanings)
        elif response.status_code == 401: logger.error(f"[Supadata] Unauthorized (401) from Supadata API for {youtube_url}. Check API Key."); return None
        elif response.status_code == 402: logger.error(f"[Supadata] Payment Required/Quota Exceeded (402) from Supadata API for {youtube_url}."); return None
        elif response.status_code == 404: logger.warning(f"[Supadata] Not Found (404) from Supadata API for {youtube_url}. Transcript might not exist."); return None
        elif response.status_code == 429: logger.error(f"[Supadata] Rate Limit Exceeded (429) from Supadata API for {youtube_url}."); return None
        elif 500 <= response.status_code < 600: logger.error(f"[Supadata] Server Error ({response.status_code}) from Supadata API for {youtube_url}. Response: {response.text}"); return None
        else: logger.error(f"[Supadata] Unexpected status code {response.status_code} from Supadata API for {youtube_url}. Response: {response.text}"); return None

    except requests.exceptions.Timeout: logger.error(f"[Supadata] Timeout error connecting to Supadata API for {youtube_url}"); return None
    except requests.exceptions.RequestException as e: logger.error(f"[Supadata] Request error connecting to Supadata API for {youtube_url}: {e}"); return None
    except Exception as e: logger.error(f"[Supadata] Unexpected error during Supadata API call for {youtube_url}: {e}", exc_info=True); return None


# --- Website Scraping Functions (Keep as they were) ---
async def get_website_content(url):
    # ... (Keep your existing BeautifulSoup scraping function) ...
    if not url: logger.error("get_website_content called with no URL"); return None
    logger.info(f"[Primary] Fetching website content for: {url}")
    try:
        headers = {
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
             'Accept-Language': 'en-US,en;q=0.5',
             'Connection': 'keep-alive', 'DNT': '1', 'Upgrade-Insecure-Requests': '1'
        }
        logger.debug(f"[Primary] Sending request to {url}")
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=25)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.debug(f"[Primary] Received response {response.status_code} from {url}")

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            logger.warning(f"[Primary] Non-HTML content type received from {url}: {content_type}")
            return None

        def parse_html(html_text):
            soup = BeautifulSoup(html_text, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "input", "iframe", "img", "svg", "link", "meta", "noscript", "figure"]): element.extract()
            main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content') or soup.find(id='main-content') or soup.find(class_='main-content') or soup.find(role='main')
            target_element = main_content if main_content else soup.body
            if not target_element: return None
            lines = [line.strip() for line in target_element.get_text(separator='\n', strip=True).splitlines() if line.strip()]
            return " ".join(lines)

        text = await asyncio.to_thread(parse_html, response.text)

        if not text:
            logger.warning(f"[Primary] Extracted text is empty after cleaning for {url}")
            return None
        logger.info(f"[Primary] Successfully scraped content for {url} (final length: {len(text)})")
        return text
    except requests.exceptions.Timeout: logger.error(f"[Primary] Timeout error scraping website: {url}"); return None
    except requests.exceptions.TooManyRedirects: logger.error(f"[Primary] Too many redirects error scraping website: {url}"); return None
    except requests.exceptions.RequestException as e: logger.error(f"[Primary] Request error scraping website {url}: {e}"); return None
    except Exception as e: logger.error(f"[Primary] Error scraping or parsing website {url}: {e}", exc_info=True); return None

async def get_website_content_via_api(url: str, api_key: str):
    # ... (Keep your existing URLToText fallback function) ...
    if not url: logger.error("[Fallback API] get_website_content_via_api called with no URL"); return None
    if not api_key: logger.error("[Fallback API] URLToText API key was not provided for this call."); return None

    logger.info(f"[Fallback API] Attempting to fetch content for: {url} using URLToText API")
    api_endpoint = "https://urltotext.com/api/v1/urltotext/"
    payload = json.dumps({"url": url, "output_format": "text", "extract_main_content": True, "render_javascript": True, "residential_proxy": False})
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}

    try:
        logger.debug(f"[Fallback API] Sending request to URLToText API for {url}")
        response = await asyncio.to_thread(requests.post, api_endpoint, headers=headers, data=payload, timeout=45)
        logger.debug(f"[Fallback API] Received status code {response.status_code} from URLToText API for {url}")

        if response.status_code == 200:
            try:
                data = response.json()
                content = data.get("data", {}).get("content")
                credits = data.get("credits_used", "N/A")
                warning = data.get("data", {}).get("warning")
                if warning: logger.warning(f"[Fallback API] URLToText API Warning for {url}: {warning}")
                if content:
                    logger.info(f"[Fallback API] Successfully fetched content via API for {url}. Length: {len(content)}. Credits used: {credits}")
                    return content.strip()
                else:
                    logger.warning(f"[Fallback API] URLToText API returned success but content was empty for {url}. Response: {data}")
                    return None
            except json.JSONDecodeError: logger.error(f"[Fallback API] Failed to decode JSON response from URLToText API for {url}. Response text: {response.text[:500]}..."); return None
            except Exception as e: logger.error(f"[Fallback API] Error processing successful URLToText API response for {url}: {e}", exc_info=True); return None
        elif response.status_code == 400: logger.error(f"[Fallback API] Bad Request (400) from URLToText API for {url}. Response: {response.text}"); return None
        elif response.status_code == 402: logger.error(f"[Fallback API] Payment Required (402) from URLToText API for {url}. Insufficient credits."); return None
        elif response.status_code == 422: logger.error(f"[Fallback API] Invalid Request / Field Error (422) from URLToText API for {url}. Response: {response.text}"); return None
        elif response.status_code == 500: logger.error(f"[Fallback API] Internal Server Error (500) from URLToText API for {url}. Response: {response.text}"); return None
        else: logger.error(f"[Fallback API] Unexpected status code {response.status_code} from URLToText API for {url}. Response: {response.text}"); return None

    except requests.exceptions.Timeout: logger.error(f"[Fallback API] Timeout error connecting to URLToText API for {url}"); return None
    except requests.exceptions.RequestException as e: logger.error(f"[Fallback API] Request error connecting to URLToText API for {url}: {e}"); return None
    except Exception as e: logger.error(f"[Fallback API] Unexpected error during URLToText API call for {url}: {e}", exc_info=True); return None

# <<< MODIFIED Gemini Summary Function >>>
async def generate_summary(text_or_url: str, summary_type: str, api_key: str) -> str:
    """Generates summary using Gemini API. Handles both text content and direct URL summarization."""
    is_direct_url_request = text_or_url.startswith('http://') or text_or_url.startswith('https://')

    if is_direct_url_request:
        logger.info(f"Generating {summary_type} summary directly from URL: {text_or_url}")
    else:
        logger.info(f"Generating {summary_type} summary. Input text length: {len(text_or_url)}")

    if not api_key:
         logger.error("Gemini API key was not provided to generate_summary.")
         return "Error: AI model configuration key is missing."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # --- Define prompts ---
        prompt_base = ""
        if summary_type == "paragraph":
            prompt_base = "You are an AI model designed to provide concise summaries using British English spellings. Your output MUST be: • Clear and simple language suitable for someone unfamiliar with the topic. • Uses British English spellings throughout. • Straightforward and understandable vocabulary; avoid complex terms. • Presented as ONE SINGLE PARAGRAPH. • No more than 85 words maximum. • Considers the entire content equally. • Uses semicolons (;) instead of em dashes (– or —)."
        else: # points summary
            prompt_base = """You are an AI model designed to provide concise summaries using British English spellings. Your output MUST strictly follow this Markdown format:
• For each distinct topic or section identified, create a heading enclosed in double asterisks (e.g., **Section Title**).
• Immediately following each heading, list key points as a bulleted list starting with a hyphen and space (`- `) on a new line.
• Bullet point text should NOT contain bold formatting.
• Use clear, simple, straightforward language (British English). Avoid complex vocabulary.
• Keep bullet points concise.
• Ensure the entire summary takes no more than two minutes to read.
• Consider the entire content, not just parts.
• Use semicolons (;) instead of em dashes (– or —)."""

        # --- Construct full prompt based on input type ---
        full_prompt = ""
        if is_direct_url_request:
            # Ask Gemini to fetch and summarize from the URL
            full_prompt = f"{prompt_base}\n\nPlease fetch the content from the following URL and provide the summary based on that content:\n{text_or_url}"
        else:
            # Summarize the provided text directly
            # Truncate long text input
            MAX_INPUT_LENGTH = 500000
            processed_text = text_or_url
            if len(processed_text) > MAX_INPUT_LENGTH:
                logger.warning(f"Input text length ({len(processed_text)}) exceeds limit ({MAX_INPUT_LENGTH}). Truncating.")
                processed_text = processed_text[:MAX_INPUT_LENGTH] + "... (Content truncated)"
            full_prompt = f"{prompt_base}\n\nHere is the text to summarise:\n{processed_text}"

        logger.debug("Sending request to Gemini model (via thread)...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = await asyncio.to_thread(
            model.generate_content,
            full_prompt,
            safety_settings=safety_settings
        )
        logger.debug("Received response from Gemini model.")

        # --- Process response (same logic as before) ---
        summary = None
        try:
            if response.parts: summary = "".join(part.text for part in response.parts)
            elif hasattr(response, 'text') and response.text: summary = response.text
        except ValueError as ve: logger.error(f"Could not extract text from Gemini response parts: {ve}. Response: {response}")
        except AttributeError as ae: logger.error(f"Unexpected response structure from Gemini: {ae}. Response: {response}")

        if summary:
            log_msg = f"Successfully generated summary from {'URL' if is_direct_url_request else 'text'}. Output length: {len(summary)}"
            logger.info(log_msg)
            return summary.strip()
        else:
            # Handle blocked or empty response
            logger.warning(f"Gemini response empty or blocked. Prompt Feedback: {response.prompt_feedback}")
            block_reason_str = "Unknown"
            safety_feedback_str = ""
            if response.prompt_feedback:
                if response.prompt_feedback.block_reason: block_reason_str = response.prompt_feedback.block_reason.name
                try:
                    if response.prompt_feedback.safety_ratings:
                        safety_feedback_str = " Safety Ratings: " + ", ".join([f"{r.category.name}: {r.probability.name}" for r in response.prompt_feedback.safety_ratings])
                except Exception as safety_ex: logger.warning(f"Could not format safety ratings: {safety_ex}")
            return f"Sorry, the summary could not be generated due to content restrictions (Reason: {block_reason_str}).{safety_feedback_str}"

    except Exception as e:
        logger.error(f"Error in generate_summary: {e}", exc_info=True)
        error_str = str(e).lower()
        # Give more specific feedback
        if "api key not valid" in error_str or "permission denied" in error_str: return "Sorry, there was an issue connecting to the AI model (API key problem)."
        elif "model not found" in error_str: return "Sorry, the specified AI model is unavailable."
        elif "rate limit" in error_str: return "Sorry, the AI model is temporarily busy. Please try again in a moment."
        elif "deadline exceeded" in error_str or "timeout" in error_str: return "Sorry, the request to the AI model timed out. Please try again."
        # Specific error if Gemini fails to process the URL
        elif is_direct_url_request and ("could not process url" in error_str or "fetching url failed" in error_str): # Hypothetical error text
             logger.warning(f"Gemini failed to process the URL directly: {text_or_url}. Error: {e}")
             return f"Sorry, the AI model could not process the content from the URL: {text_or_url}"
        return "Sorry, an unexpected error occurred while generating the summary with the AI model."


# --- Telegram Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (Keep as is) ...
    user = update.effective_user
    logger.info(f"User {user.id} ({user.username or 'NoUsername'}) used /start.")
    mention = user.mention_html() if user.username else user.first_name
    await update.message.reply_html(f"👋 Hello {mention}! I can summarize YouTube links or website URLs.\n\nJust send me a link anytime!",)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (Keep as is) ...
    await update.message.reply_text(
        "🔍 **How to use this bot:**\n\n"
        "1. Send me any YouTube video link or website URL.\n"
        "2. I'll ask you how you want it summarized (paragraph or points).\n"
        "3. Click the button for your choice.\n"
        "4. Wait for the summary!\n\n"
        "For YouTube, I'll try a fast API first. If that fails, I'll ask the AI to summarize the video directly from the link.\n"
        "For websites, I'll try scraping directly, with a fallback API if needed.\n\n"
        "**Commands:**\n"
        "/start - Display welcome message\n"
        "/help - Show this help message",
        parse_mode='Markdown'
        )

async def handle_potential_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (Keep as is) ...
    if not update.message or not update.message.text: return
    url = update.message.text.strip()
    user = update.effective_user
    logger.info(f"User {user.id} sent potential URL: {url}")
    if not (url.startswith('http://') or url.startswith('https://')) or '.' not in url[8:]:
        logger.debug(f"Ignoring non-URL message from user {user.id}: {url}")
        return
    context.user_data['url_to_summarize'] = url
    logger.debug(f"Stored URL for user {user.id} in user_data")
    keyboard = [[InlineKeyboardButton("Paragraph Summary", callback_data="paragraph"), InlineKeyboardButton("Points Summary", callback_data="points")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Okay, I see this link:\n`{url}`\n\nHow would you like it summarized?", reply_markup=reply_markup, disable_web_page_preview=True, parse_mode='Markdown')

# <<< MODIFIED Callback Handler >>>
async def handle_summary_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query: return
    await query.answer()

    summary_type = query.data
    user = update.effective_user or query.from_user
    chat_id = query.message.chat_id

    logger.info(f"User {user.id} chose '{summary_type}' summary. Checking context for URL.")

    url = context.user_data.pop('url_to_summarize', None)

    if not url:
         logger.warning(f"User {user.id} pressed button, but NO URL found (chat_id: {chat_id}).")
         try: await query.edit_message_text(text="Sorry, I couldn't find the URL. Please send the link again.")
         except Exception as edit_err: logger.warning(f"Failed to edit message: {edit_err}"); await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't find the URL. Please send the link again.")
         return

    logger.debug(f"Retrieved URL '{url}' from user_data for user {user.id}")

    # --- Get API keys ---
    current_gemini_key = os.environ.get(GEMINI_API_KEY_ENV)
    current_urltotext_key = os.environ.get(URLTOTEXT_API_KEY_ENV)
    current_supadata_key = os.environ.get(SUPADATA_API_KEY_ENV) # <<< Get Supadata key

    # Essential key check
    if not current_gemini_key:
        logger.error(f"CRITICAL: {GEMINI_API_KEY_ENV} not found.")
        await context.bot.send_message(chat_id=chat_id, text=f"Error: AI model config ({GEMINI_API_KEY_ENV}) missing. Contact admin.")
        try: await query.delete_message()
        except Exception: pass
        return

    # Optional key warnings
    if not current_urltotext_key: logger.warning(f"{URLTOTEXT_API_KEY_ENV} not found. Website fallback API disabled.")
    if not current_supadata_key: logger.warning(f"{SUPADATA_API_KEY_ENV} not found. Primary YouTube transcript API disabled, will fallback to Gemini URL.")

    # --- Start Processing ---
    processing_message_text = f"Got it! Generating '{summary_type}' summary for:\n`{url}`\n\nWorking on it..."
    message_to_delete_later = None
    try: await query.edit_message_text(text=processing_message_text, parse_mode='Markdown', disable_web_page_preview=True)
    except Exception as e:
        logger.warning(f"Could not edit original message: {e}. Sending new status message.")
        try: message_to_delete_later = await context.bot.send_message(chat_id=chat_id, text=processing_message_text, parse_mode='Markdown', disable_web_page_preview=True)
        except Exception as send_err: logger.error(f"Fatal: Failed to send status message: {send_err}"); return

    content_to_summarize = None # Will hold text transcript OR the URL for Gemini fallback
    use_gemini_directly_with_url = False
    user_feedback_message = None
    success = False

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        is_youtube = is_youtube_url(url)

        if is_youtube:
            logger.info(f"Processing YouTube URL: {url}")
            if current_supadata_key:
                # --- Try Supadata First ---
                transcript = await get_youtube_transcript_via_supadata(url, current_supadata_key)
                if transcript:
                    logger.info("Supadata transcript obtained successfully.")
                    content_to_summarize = transcript
                else:
                    logger.warning("Supadata failed or returned no transcript. Falling back to Gemini direct URL summarization.")
                    use_gemini_directly_with_url = True
                    content_to_summarize = url # Pass the URL itself to generate_summary
            else:
                # --- No Supadata Key, Go Directly to Gemini Fallback ---
                logger.warning(f"{SUPADATA_API_KEY_ENV} missing. Using Gemini direct URL summarization for YouTube.")
                use_gemini_directly_with_url = True
                content_to_summarize = url # Pass the URL itself

        else: # It's a website URL
            logger.info(f"Processing Website URL: {url}")
            # --- Try primary scraping method ---
            content = await get_website_content(url)
            if content:
                logger.info(f"Primary website scraping successful for {url}.")
                content_to_summarize = content
            else:
                logger.warning(f"Primary scraping failed for {url}. Attempting fallback API.")
                # --- Fallback Logic for Websites ---
                if current_urltotext_key:
                    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
                    content = await get_website_content_via_api(url, current_urltotext_key)
                    if content:
                        logger.info(f"Fallback URLToText API scraping successful for {url}.")
                        content_to_summarize = content
                    else:
                        user_feedback_message = "Sorry, I couldn't fetch content from that website using either method (blocked/inaccessible/empty?)."
                        logger.error(f"Both primary and fallback website methods failed for {url}.")
                else:
                    user_feedback_message = "Sorry, I couldn't fetch content from that website (primary method failed). The fallback method is not configured."
                    logger.warning(f"Primary scraping failed for {url}, and {URLTOTEXT_API_KEY_ENV} is missing.")

        # --- Generate Summary if Content or URL is Ready ---
        if content_to_summarize:
            log_action = "Summarizing transcript" if (is_youtube and not use_gemini_directly_with_url) else \
                         "Asking AI to summarize YouTube URL directly" if use_gemini_directly_with_url else \
                         "Summarizing website content"
            logger.info(f"{log_action} using Gemini.")
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            # Pass either the transcript text or the URL to generate_summary
            summary = await generate_summary(content_to_summarize, summary_type, current_gemini_key)

            if summary.startswith("Error:") or summary.startswith("Sorry,"):
                 user_feedback_message = summary # Use the error message from Gemini
                 logger.warning(f"Summary generation failed or returned error: {summary}")
            else:
                 await context.bot.send_message(chat_id=chat_id, text=summary, parse_mode='Markdown', disable_web_page_preview=True)
                 success = True
                 user_feedback_message = None # Clear previous failures

        # --- Send Feedback if any step failed and no summary was sent ---
        elif not user_feedback_message: # If content_to_summarize was None and no specific error was set
            user_feedback_message = "Sorry, I couldn't retrieve any content to summarize for that link."

        if user_feedback_message and not success:
            await context.bot.send_message(chat_id=chat_id, text=user_feedback_message)

    except Exception as e:
        logger.error(f"Unexpected error in callback for URL {url}: {e}", exc_info=True)
        try: await context.bot.send_message(chat_id=chat_id, text="Oops! Something went wrong processing your request.")
        except Exception as final_err: logger.error(f"Failed to send final error message: {final_err}")
    finally:
        # Clean up status message
        try:
             if message_to_delete_later: await context.bot.delete_message(chat_id=chat_id, message_id=message_to_delete_later.message_id)
             elif query: await query.delete_message()
        except Exception as del_e: logger.warning(f"Could not delete status/button message: {del_e}")

# --- Error handler (Keep as is) ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

# <<< MODIFIED Main Function >>>
def main() -> None:
    """Runs the Telegram bot."""
    logger.info("Starting bot...")
    token = os.environ.get(TELEGRAM_TOKEN_ENV)
    if not token: logger.critical(f"CRITICAL ERROR: {TELEGRAM_TOKEN_ENV} not set."); sys.exit("Error: Bot token missing.")

    # Check for keys at startup
    if not os.environ.get(GEMINI_API_KEY_ENV): logger.warning(f"WARNING: {GEMINI_API_KEY_ENV} not set. Summarization will fail.")
    else: logger.info(f"{GEMINI_API_KEY_ENV} found.")

    if not os.environ.get(URLTOTEXT_API_KEY_ENV): logger.warning(f"WARNING: {URLTOTEXT_API_KEY_ENV} not set. Website fallback API disabled.")
    else: logger.info(f"{URLTOTEXT_API_KEY_ENV} found.")

    if not os.environ.get(SUPADATA_API_KEY_ENV): logger.warning(f"WARNING: {SUPADATA_API_KEY_ENV} not set. Primary YouTube API disabled; will fallback to Gemini URL method.") # <<< Check Supadata Key
    else: logger.info(f"{SUPADATA_API_KEY_ENV} found.")

    application = Application.builder().token(token).build()

    # Register Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_potential_url))
    application.add_handler(CallbackQueryHandler(handle_summary_type_callback))
    application.add_error_handler(error_handler)

    logger.info("Bot is configured. Starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot polling stopped.")

if __name__ == '__main__':
    main()
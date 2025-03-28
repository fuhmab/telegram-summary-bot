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
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
# Import types for File API
import google.generativeai.types as genai_types

# --- Environment Variable Names (Constants) ---
TELEGRAM_TOKEN_ENV = "TELEGRAM_TOKEN"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
URLTOTEXT_API_KEY_ENV = "URLTOTEXT_API_KEY"
SUPADATA_API_KEY_ENV = "SUPADATA_API_KEY"

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def is_youtube_url(url):
    """Checks if a string is a valid YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)([\w-]{11})'
    return bool(re.search(youtube_regex, url))

# --- Content Fetching Functions ---

async def get_youtube_transcript_via_supadata(youtube_url: str, api_key: str):
    """Attempts to fetch YouTube transcript using the Supadata API."""
    if not youtube_url: logger.error("[Supadata] get_youtube_transcript_via_supadata called with no URL"); return None
    if not api_key: logger.error("[Supadata] Supadata API key was not provided."); return None

    logger.info(f"[Supadata] Attempting to fetch transcript for: {youtube_url}")
    encoded_url = urllib.parse.quote(youtube_url, safe='')
    api_endpoint = f"https://api.supadata.ai/v1/youtube/transcript?url={encoded_url}"
    headers = {"x-api-key": api_key, "Accept": "application/json"}

    try:
        logger.debug(f"[Supadata] Sending request to Supadata API: {api_endpoint}")
        response = await asyncio.to_thread(requests.get, api_endpoint, headers=headers, timeout=45)
        logger.debug(f"[Supadata] Received status code {response.status_code} from Supadata API for {youtube_url}")

        if response.status_code == 200:
            try:
                data = response.json()
                content_list = data.get("content")
                if isinstance(content_list, list) and content_list:
                    texts = [item.get('text', '') for item in content_list if isinstance(item, dict)]
                    full_transcript = " ".join(filter(None, texts)).strip()
                    if full_transcript:
                        logger.info(f"[Supadata] Successfully fetched and assembled transcript. Length: {len(full_transcript)}")
                        return full_transcript
                    else: logger.warning(f"[Supadata] Extracted text was empty. Response: {data}"); return None
                else: logger.warning(f"[Supadata] 'content' list missing or empty. Response: {data}"); return None
            except json.JSONDecodeError: logger.error(f"[Supadata] Failed to decode JSON response: {response.text[:500]}..."); return None
            except Exception as e: logger.error(f"[Supadata] Error processing Supadata response: {e}", exc_info=True); return None
        elif response.status_code == 401: logger.error(f"[Supadata] Unauthorized (401). Check API Key."); return None
        elif response.status_code == 402: logger.error(f"[Supadata] Payment Required/Quota Exceeded (402)."); return None
        elif response.status_code == 404: logger.warning(f"[Supadata] Not Found (404). Transcript might not exist."); return None
        elif response.status_code == 429: logger.error(f"[Supadata] Rate Limit Exceeded (429)."); return None
        elif 500 <= response.status_code < 600: logger.error(f"[Supadata] Server Error ({response.status_code}): {response.text}"); return None
        else: logger.error(f"[Supadata] Unexpected status code {response.status_code}: {response.text}"); return None

    except requests.exceptions.Timeout: logger.error(f"[Supadata] Timeout error connecting."); return None
    except requests.exceptions.RequestException as e: logger.error(f"[Supadata] Request error: {e}"); return None
    except Exception as e: logger.error(f"[Supadata] Unexpected error: {e}", exc_info=True); return None

async def get_website_content(url):
    """Attempts to scrape website content using BeautifulSoup (Primary Method)."""
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
    """Attempts to fetch website content using the URLToText API (Fallback Method)."""
    if not url: logger.error("[Fallback API] get_website_content_via_api called with no URL"); return None
    if not api_key: logger.error("[Fallback API] URLToText API key was not provided."); return None

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
            except json.JSONDecodeError: logger.error(f"[Fallback API] Failed to decode JSON response: {response.text[:500]}..."); return None
            except Exception as e: logger.error(f"[Fallback API] Error processing successful URLToText API response for {url}: {e}", exc_info=True); return None
        elif response.status_code == 400: logger.error(f"[Fallback API] Bad Request (400) from URLToText API for {url}. Response: {response.text}"); return None
        elif response.status_code == 402: logger.error(f"[Fallback API] Payment Required (402) from URLToText API for {url}. Insufficient credits."); return None
        elif response.status_code == 422: logger.error(f"[Fallback API] Invalid Request / Field Error (422) from URLToText API for {url}. Response: {response.text}"); return None
        elif response.status_code == 500: logger.error(f"[Fallback API] Internal Server Error (500) from URLToText API for {url}. Response: {response.text}"); return None
        else: logger.error(f"[Fallback API] Unexpected status code {response.status_code} from URLToText API for {url}. Response: {response.text}"); return None

    except requests.exceptions.Timeout: logger.error(f"[Fallback API] Timeout error connecting to URLToText API for {url}"); return None
    except requests.exceptions.RequestException as e: logger.error(f"[Fallback API] Request error connecting to URLToText API for {url}: {e}"); return None
    except Exception as e: logger.error(f"[Fallback API] Unexpected error during URLToText API call for {url}: {e}", exc_info=True); return None


async def generate_summary(text_or_youtube_url: str, summary_type: str, api_key: str) -> str:
    """Generates summary using Gemini API. Handles text content OR direct YouTube URL input."""
    is_youtube_url_input = is_youtube_url(text_or_youtube_url)
    log_input_type = "YouTube URL" if is_youtube_url_input else "text content"
    log_input_value = text_or_youtube_url if is_youtube_url_input else f"Length: {len(text_or_youtube_url)}"
    logger.info(f"Generating {summary_type} summary from {log_input_type} ({log_input_value})")

    if not api_key:
         logger.error("Gemini API key was not provided to generate_summary.")
         return "Error: AI model configuration key is missing."

    try:
        genai.configure(api_key=api_key)
        # Using the requested experimental model
        model_name = 'gemini-1.5-pro-exp-03-25'
        logger.warning(f"Using EXPERIMENTAL Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)

        prompt_base = ""
        if summary_type == "paragraph":
            # Keep your detailed paragraph prompt instructions here
            prompt_base = "You are an AI model designed to provide concise summaries using British English spellings. Your output MUST be: • Clear and simple language suitable for someone unfamiliar with the topic. • Uses British English spellings throughout. • Straightforward and understandable vocabulary; avoid complex terms. • Presented as ONE SINGLE PARAGRAPH. • No more than 85 words maximum. • Considers the entire content equally. • Uses semicolons (;) instead of em dashes (– or —)."
        else: # points summary
             # Keep your detailed points prompt instructions here
            prompt_base = """You are an AI model designed to provide concise summaries using British English spellings. Your output MUST strictly follow this Markdown format:
• For each distinct topic or section identified, create a heading enclosed in double asterisks (e.g., **Section Title**).
• Immediately following each heading, list key points as a bulleted list starting with a hyphen and space (`- `) on a new line.
• Bullet point text should NOT contain bold formatting.
• Use clear, simple, straightforward language (British English). Avoid complex vocabulary.
• Keep bullet points concise.
• Ensure the entire summary takes no more than two minutes to read.
• Consider the entire content, not just parts.
• Use semicolons (;) instead of em dashes (– or —)."""

        api_contents = None
        text_prompt = f"Please summarize the key points of the provided content according to the following instructions:\n\n{prompt_base}"

        if is_youtube_url_input:
            logger.info(f"Constructing multi-part request for Gemini with YouTube URL: {text_or_youtube_url}")
            api_contents = [
                text_prompt,
                genai_types.Part(
                    file_data=genai_types.FileData(file_uri=text_or_youtube_url)
                )
            ]
        else:
            MAX_INPUT_LENGTH = 900000
            processed_text = text_or_youtube_url
            if len(processed_text) > MAX_INPUT_LENGTH:
                logger.warning(f"Input text length ({len(processed_text)}) exceeds limit ({MAX_INPUT_LENGTH}). Truncating.")
                processed_text = processed_text[:MAX_INPUT_LENGTH] + "... (Content truncated)"
            api_contents = f"{text_prompt}\n\nHere is the text to summarise:\n{processed_text}"

        logger.debug(f"Sending request to Gemini model '{model_name}'...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = await asyncio.to_thread(
            model.generate_content,
            api_contents,
            safety_settings=safety_settings
        )
        logger.debug("Received response from Gemini model.")

        summary = None
        try:
            if response.parts: summary = "".join(part.text for part in response.parts)
            elif hasattr(response, 'text') and response.text: summary = response.text
        except (ValueError, AttributeError) as e: logger.error(f"Could not extract text from Gemini response: {e}. Response: {response}")

        if summary:
            logger.info(f"Successfully generated summary from {log_input_type}. Output length: {len(summary)}")
            return summary.strip()
        else:
            logger.warning(f"Gemini response empty or blocked. Prompt Feedback: {response.prompt_feedback}")
            block_reason_str = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason_str = response.prompt_feedback.block_reason.name
            return f"Sorry, the summary could not be generated due to content restrictions (Reason: {block_reason_str})."

    except Exception as e:
        logger.error(f"Error in generate_summary: {e}", exc_info=True)
        error_str = str(e).lower()
        # Specific errors using isinstance might be better if library raises distinct exceptions
        if "permission denied" in error_str and "file" in error_str and is_youtube_url_input:
             logger.error(f"Gemini Permission Denied for YouTube URL: {text_or_youtube_url}. Is it public?")
             return f"Sorry, the AI model couldn't access the YouTube URL '{text_or_youtube_url}'. It might be private, unlisted, or restricted."
        if "invalid argument" in error_str and "uri" in error_str and is_youtube_url_input:
             logger.error(f"Gemini Invalid Argument for YouTube URL: {text_or_youtube_url}. Format/Length issue?")
             return f"Sorry, there was an issue processing the YouTube URL '{text_or_youtube_url}' with the AI model (invalid format or length?)."
        # General errors
        if "api key not valid" in error_str: return "Sorry, AI connection issue (API key problem)."
        if "model not found" in error_str: return f"Sorry, the AI model '{model_name}' is unavailable."
        if "rate limit" in error_str: return "Sorry, the AI model is busy. Please try again."
        if "deadline exceeded" in error_str or "timeout" in error_str: return "Sorry, AI request timed out."
        return "Sorry, an unexpected error occurred generating the summary."


# --- Telegram Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # <<< Indentation Fix Applied Here >>>
    user = update.effective_user
    logger.info(f"User {user.id} ({user.username or 'NoUsername'}) used /start.")
    mention = user.mention_html() if user.username else user.first_name
    await update.message.reply_html(f"👋 Hello {mention}! I can summarize YouTube links or website URLs.\n\nJust send me a link anytime!",)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # <<< Indentation Fix Applied Here >>>
    await update.message.reply_text(
        "🔍 **How to use this bot:**\n\n"
        "1. Send a YouTube video link or website URL.\n"
        "2. Choose 'Paragraph' or 'Points' summary.\n"
        "3. Wait for the summary!\n\n"
        "For YouTube, I try a fast transcript API first. If that fails, the AI model will process the YouTube link directly.\n"
        "For websites, I try direct scraping, then a fallback API if needed.\n\n"
        "**Commands:**\n"
        "/start, /help",
        parse_mode='Markdown'
        )

async def handle_potential_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # <<< Indentation Fix Applied Here >>>
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


async def handle_summary_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query: return
    await query.answer()

    summary_type = query.data
    user = update.effective_user or query.from_user
    chat_id = query.message.chat_id
    logger.info(f"User {user.id} chose '{summary_type}'. Getting URL.")
    url = context.user_data.pop('url_to_summarize', None)
    if not url:
        logger.warning(f"No URL found for {user.id}.")
        try: await query.edit_message_text(text="Sorry, I couldn't find the URL. Please send the link again.")
        except Exception: await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't find the URL. Please send the link again.")
        return
    logger.debug(f"Retrieved URL '{url}' for {user.id}")

    # --- Get API keys ---
    current_gemini_key = os.environ.get(GEMINI_API_KEY_ENV)
    current_urltotext_key = os.environ.get(URLTOTEXT_API_KEY_ENV)
    current_supadata_key = os.environ.get(SUPADATA_API_KEY_ENV)
    if not current_gemini_key:
        logger.error(f"CRITICAL: {GEMINI_API_KEY_ENV} not found.")
        await context.bot.send_message(chat_id=chat_id, text=f"Error: AI model config ({GEMINI_API_KEY_ENV}) missing. Contact admin.")
        try: await query.delete_message()
        except Exception: pass
        return
    if not current_urltotext_key: logger.warning(f"{URLTOTEXT_API_KEY_ENV} not found.")
    if not current_supadata_key: logger.warning(f"{SUPADATA_API_KEY_ENV} not found.")

    # --- Start Processing ---
    processing_message_text = f"Got it! Generating '{summary_type}' summary for:\n`{url}`\n\nWorking on it..."
    message_to_delete_later = None
    try:
        await query.edit_message_text(text=processing_message_text, parse_mode='Markdown', disable_web_page_preview=True)
    except Exception as e:
        logger.warning(f"Could not edit original message: {e}. Sending new.")
        try: message_to_delete_later = await context.bot.send_message(chat_id=chat_id, text=processing_message_text, parse_mode='Markdown', disable_web_page_preview=True)
        except Exception as send_err: logger.error(f"Fatal: Failed to send status message: {send_err}"); return

    content_to_summarize = None
    user_feedback_message = None
    success = False

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        is_youtube = is_youtube_url(url)

        if is_youtube:
            logger.info(f"Processing YouTube URL: {url}")
            transcript = None
            if current_supadata_key:
                transcript = await get_youtube_transcript_via_supadata(url, current_supadata_key)

            if transcript:
                logger.info("Supadata transcript obtained.")
                content_to_summarize = transcript
            else:
                logger.warning("Supadata failed or key missing. Using Gemini direct YouTube URL processing.")
                content_to_summarize = url # Pass URL for Gemini File API
        else: # It's a website URL
             logger.info(f"Processing Website URL: {url}")
             content = await get_website_content(url)
             if content: content_to_summarize = content
             else:
                 logger.warning(f"Primary scraping failed for {url}. Trying fallback.")
                 if current_urltotext_key:
                     await context.bot.send_chat_action(chat_id=chat_id, action='typing')
                     content = await get_website_content_via_api(url, current_urltotext_key)
                     if content:
                         logger.info("Fallback URLToText API scraping successful.")
                         content_to_summarize = content
                     else:
                         user_feedback_message = "Sorry, I couldn't fetch content from that website using either method."
                         logger.error(f"Both primary and fallback website methods failed for {url}.")
                 else:
                     user_feedback_message = "Sorry, couldn't scrape website content (no fallback configured)."
                     logger.warning(f"Primary scraping failed for {url}, and {URLTOTEXT_API_KEY_ENV} is missing.")

        # --- Generate Summary if Content or YT URL is Ready ---
        if content_to_summarize:
            log_action = "Summarizing content" if not (is_youtube and content_to_summarize == url) else \
                         "Asking AI to summarize YouTube URL directly"
            logger.info(f"{log_action} using Gemini.")
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            summary = await generate_summary(content_to_summarize, summary_type, current_gemini_key)

            if summary.startswith("Error:") or summary.startswith("Sorry,"):
                 user_feedback_message = summary
                 logger.warning(f"Summary generation failed/returned error: {summary}")
            else:
                 await context.bot.send_message(chat_id=chat_id, text=summary, parse_mode='Markdown', disable_web_page_preview=True)
                 success = True
                 user_feedback_message = None

        # --- Send Feedback if necessary ---
        elif not user_feedback_message: # If no content and no specific error set
             user_feedback_message = "Sorry, couldn't retrieve any content to summarize for that link."

        if user_feedback_message and not success:
            await context.bot.send_message(chat_id=chat_id, text=user_feedback_message)

    except Exception as e:
        logger.error(f"Unexpected error in callback: {e}", exc_info=True)
        try: await context.bot.send_message(chat_id=chat_id, text="Oops! Something went wrong processing your request.")
        except Exception as final_err: logger.error(f"Failed to send final error message: {final_err}")
    finally:
        # Clean up status message
        try:
             if message_to_delete_later: await context.bot.delete_message(chat_id=chat_id, message_id=message_to_delete_later.message_id)
             elif query: await query.delete_message()
        except Exception as del_e: logger.warning(f"Could not delete status/button message: {del_e}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)


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
    if not os.environ.get(SUPADATA_API_KEY_ENV): logger.warning(f"WARNING: {SUPADATA_API_KEY_ENV} not set. Primary YouTube API disabled; will fallback to Gemini.")
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
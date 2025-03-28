# bot.py
import os
import re
import logging
import asyncio
import json
import sys
import urllib.parse

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
# <<< NEW IMPORT for FileData >>>
import google.generativeai.types as genai_types # Use alias to avoid conflicts

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
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)([\w-]{11})'
    return bool(re.search(youtube_regex, url))

# --- Content Fetching Functions ---

# Supadata function (with previous fix)
async def get_youtube_transcript_via_supadata(youtube_url: str, api_key: str):
    # ... (Keep the corrected Supadata function from the previous message) ...
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

# Website scraping functions (Keep as they were)
async def get_website_content(url):
    # ... (Keep existing function) ...
     if not url: logger.error("get_website_content called with no URL"); return None
     logger.info(f"[Primary] Fetching website content for: {url}")
     try:
         headers = {'User-Agent': 'Mozilla/5.0 ...'} # Use your actual headers
         response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=25)
         response.raise_for_status()
         # ... rest of scraping logic ...
         # ... return text or None ...
     except Exception as e: logger.error(f"[Primary] Error scraping {url}: {e}"); return None

async def get_website_content_via_api(url: str, api_key: str):
    # ... (Keep existing function) ...
    if not url: logger.error("[Fallback API] get_website_content_via_api called with no URL"); return None
    if not api_key: logger.error("[Fallback API] URLToText API key was not provided."); return None
    # ... rest of URLToText API call logic ...
    # ... return text or None ...


# <<< NEW MODIFIED Gemini Summary Function >>>
async def generate_summary(text_or_youtube_url: str, summary_type: str, api_key: str) -> str:
    """Generates summary using Gemini API. Handles text content OR direct YouTube URL input."""
    # Detect if input is a YouTube URL for direct processing
    is_youtube_url_input = is_youtube_url(text_or_youtube_url)

    log_input_type = "YouTube URL" if is_youtube_url_input else "text content"
    log_input_value = text_or_youtube_url if is_youtube_url_input else f"Length: {len(text_or_youtube_url)}"
    logger.info(f"Generating {summary_type} summary from {log_input_type} ({log_input_value})")

    if not api_key:
         logger.error("Gemini API key was not provided to generate_summary.")
         return "Error: AI model configuration key is missing."

    try:
        # Configure API Key
        genai.configure(api_key=api_key)

        # --- Use the requested EXPERIMENTAL model ---
        # NOTE: Experimental models might change or be removed without notice.
        # Consider falling back to 'gemini-1.5-pro-latest' or 'gemini-1.5-flash-latest' if this causes issues.
        model_name = 'gemini-1.5-pro-exp-03-25' # <<< USER REQUESTED MODEL
        logger.warning(f"Using EXPERIMENTAL Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)

        # --- Define base prompt for summarization format ---
        prompt_base = ""
        if summary_type == "paragraph":
            prompt_base = "You are an AI model... Provide a concise ONE SINGLE PARAGRAPH summary (max 85 words) using British English spellings, simple language, and semicolons instead of em dashes." # Keep your full detailed prompt
        else: # points summary
            prompt_base = """You are an AI model... Provide a summary in Markdown format: **Heading**\n- Bullet point (British English, simple, no bold)\n- Another bullet point... Use semicolons instead of em dashes.""" # Keep your full detailed prompt

        # --- Construct the 'contents' argument for the API call ---
        api_contents = None
        text_prompt = f"Please summarize the key points of the provided content according to the following instructions:\n\n{prompt_base}"

        if is_youtube_url_input:
            logger.info(f"Constructing multi-part request for Gemini with YouTube URL: {text_or_youtube_url}")
            # Create structured input with text prompt and YouTube file URI
            api_contents = [
                text_prompt, # The text part containing instructions
                # The video part using FileData URI (as per new docs)
                genai_types.Part(
                    file_data=genai_types.FileData(file_uri=text_or_youtube_url)
                )
            ]
        else:
            # Input is plain text (from Supadata or website scraping)
            # Truncate long text input
            MAX_INPUT_LENGTH = 900000 # Adjusted slightly for potentially larger context windows, but still good practice
            processed_text = text_or_youtube_url
            if len(processed_text) > MAX_INPUT_LENGTH:
                logger.warning(f"Input text length ({len(processed_text)}) exceeds limit ({MAX_INPUT_LENGTH}). Truncating.")
                processed_text = processed_text[:MAX_INPUT_LENGTH] + "... (Content truncated)"

            # Use simple string input for text summarization
            api_contents = f"{text_prompt}\n\nHere is the text to summarise:\n{processed_text}"


        # --- Make the API call ---
        logger.debug(f"Sending request to Gemini model '{model_name}'...")
        safety_settings = [ # Keep safety settings
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # Use generate_content with either the string or the list contents
        response = await asyncio.to_thread(
            model.generate_content,
            api_contents, # Pass either the string or the list
            safety_settings=safety_settings
            # generation_config={"response_mime_type": "text/plain"} # Optional: Ensure plain text output
        )
        logger.debug("Received response from Gemini model.")

        # --- Process response (same logic as before) ---
        summary = None
        try:
            if response.parts: summary = "".join(part.text for part in response.parts)
            elif hasattr(response, 'text') and response.text: summary = response.text
        except (ValueError, AttributeError) as e: logger.error(f"Could not extract text from Gemini response: {e}. Response: {response}")

        if summary:
            logger.info(f"Successfully generated summary from {log_input_type}. Output length: {len(summary)}")
            return summary.strip()
        else:
            # Handle blocked or empty response
            logger.warning(f"Gemini response empty or blocked. Prompt Feedback: {response.prompt_feedback}")
            # ... (keep block reason reporting logic) ...
            block_reason_str = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason_str = response.prompt_feedback.block_reason.name
            return f"Sorry, the summary could not be generated due to content restrictions (Reason: {block_reason_str})."

    # --- Error Handling (Keep detailed checks) ---
    except Exception as e:
        logger.error(f"Error in generate_summary: {e}", exc_info=True)
        error_str = str(e).lower()
        # Specific errors for File API if they arise (check library's error types)
        if isinstance(e, genai_types.google.api_core.exceptions.PermissionDenied) and is_youtube_url_input:
             logger.error(f"Gemini Permission Denied for YouTube URL: {text_or_youtube_url}. Is it public?")
             return f"Sorry, the AI model couldn't access the YouTube URL '{text_or_youtube_url}'. It might be private, unlisted, or restricted."
        if isinstance(e, genai_types.google.api_core.exceptions.InvalidArgument) and is_youtube_url_input:
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
    # ... (Keep as is) ...

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (Update help text slightly if desired, e.g., mentioning Gemini processes YT links directly) ...
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
    # ... (Keep as is) ...

# <<< MODIFIED Callback Handler (Re-enabled Fallback Logic) >>>
async def handle_summary_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    # ... (keep initial query/user/chat_id/url retrieval logic) ...
    if not query: return
    await query.answer()
    summary_type = query.data; user = update.effective_user or query.from_user; chat_id = query.message.chat_id
    logger.info(f"User {user.id} chose '{summary_type}'. Getting URL.")
    url = context.user_data.pop('url_to_summarize', None)
    if not url: logger.warning(f"No URL found for {user.id}."); # ... keep error handling ...
        return
    logger.debug(f"Retrieved URL '{url}' for {user.id}")

    # --- Get API keys ---
    current_gemini_key = os.environ.get(GEMINI_API_KEY_ENV)
    current_urltotext_key = os.environ.get(URLTOTEXT_API_KEY_ENV)
    current_supadata_key = os.environ.get(SUPADATA_API_KEY_ENV)
    if not current_gemini_key: # ... keep Gemini key error check ...
        return
    if not current_urltotext_key: logger.warning(f"{URLTOTEXT_API_KEY_ENV} not found.")
    if not current_supadata_key: logger.warning(f"{SUPADATA_API_KEY_ENV} not found.")

    # --- Start Processing ---
    processing_message_text = f"Got it! Generating '{summary_type}' summary for:\n`{url}`\n\nWorking on it..."
    # ... (keep message editing/sending logic) ...

    content_to_summarize = None # Will hold text OR the youtube URL for Gemini
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
                # --- Supadata failed or key missing, FALLBACK TO GEMINI DIRECT URL ---
                logger.warning("Supadata failed or key missing. Using Gemini direct YouTube URL processing.")
                # Pass the URL itself to generate_summary
                content_to_summarize = url
        else: # It's a website URL
            # ... (Keep your existing website scraping logic) ...
             logger.info(f"Processing Website URL: {url}")
             content = await get_website_content(url)
             if content: content_to_summarize = content
             else:
                 logger.warning(f"Primary scraping failed for {url}. Trying fallback.")
                 if current_urltotext_key:
                     # ... (call get_website_content_via_api) ...
                     # ... (set content_to_summarize or user_feedback_message) ...
                 else: user_feedback_message = "Sorry, couldn't scrape website content (no fallback configured)."


        # --- Generate Summary if Content or YT URL is Ready ---
        if content_to_summarize:
            log_action = "Summarizing content" if not (is_youtube and content_to_summarize == url) else \
                         "Asking AI to summarize YouTube URL directly"
            logger.info(f"{log_action} using Gemini.")
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            # Pass either the text or the YouTube URL
            summary = await generate_summary(content_to_summarize, summary_type, current_gemini_key)

            if summary.startswith("Error:") or summary.startswith("Sorry,"):
                 user_feedback_message = summary
                 logger.warning(f"Summary generation failed/returned error: {summary}")
            else:
                 # ... (send successful summary) ...
                 success = True; user_feedback_message = None

        # --- Send Feedback if necessary ---
        elif not user_feedback_message: user_feedback_message = "Sorry, couldn't retrieve content."
        if user_feedback_message and not success: await context.bot.send_message(chat_id=chat_id, text=user_feedback_message)

    # ... (keep except Exception and finally blocks) ...
    except Exception as e: logger.error(f"Unexpected error in callback: {e}", exc_info=True); # ... handle ...
    finally: # ... clean up messages ...


# --- Error handler & Main function ---
# ... (Keep error_handler and main function as they were) ...

if __name__ == '__main__':
    main()
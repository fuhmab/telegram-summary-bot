# bot.py
import os
import re
import logging
import asyncio
import json
import sys # To exit if token is missing

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
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
# Import Gemini library
import google.generativeai as genai

# --- Environment Variable Names (Constants) ---
# Use these names when setting up Environment Variables in Render
TELEGRAM_TOKEN_ENV = "TELEGRAM_TOKEN"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
URLTOTEXT_API_KEY_ENV = "URLTOTEXT_API_KEY" # Key for the fallback API

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO # Render captures INFO level logs
)
# Reduce noisy logging from underlying HTTP library
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def is_youtube_url(url):
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)([\w-]{11})'
    return bool(re.search(youtube_regex, url))

def extract_youtube_id(url):
    youtube_id_regex = r'(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})'
    match = re.search(youtube_id_regex, url)
    if match:
        return match.group(1)
    logger.warning(f"Could not extract YouTube ID from URL: {url}")
    return None

# --- Content Fetching Functions ---
async def get_youtube_transcript(video_id):
    if not video_id: logger.error("get_youtube_transcript called with no video_id"); return None
    logger.info(f"Fetching transcript for video ID: {video_id}")
    try:
        # Run the synchronous library call in a separate thread
        transcript_list = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id, languages=['en', 'en-GB'])
        if not transcript_list: logger.warning(f"Transcript list empty for {video_id}"); return None
        transcript_text = " ".join([item['text'] for item in transcript_list if 'text' in item])
        if not transcript_text: logger.warning(f"Joined transcript text is empty for {video_id}"); return None
        logger.info(f"Successfully fetched transcript for {video_id} (length: {len(transcript_text)})")
        return transcript_text
    except Exception as e:
        logger.error(f"Error getting YouTube transcript for {video_id}: {e}")
        if "No transcript found" in str(e): logger.warning(f"No transcript found for {video_id}. May be unavailable/private.")
        elif "disabled" in str(e): logger.warning(f"Transcripts disabled for {video_id}.")
        return None

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
        # Run the synchronous requests.get in a separate thread
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=25)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.debug(f"[Primary] Received response {response.status_code} from {url}")

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            logger.warning(f"[Primary] Non-HTML content type received from {url}: {content_type}")
            return None

        # Run synchronous BeautifulSoup parsing in a separate thread
        def parse_html(html_text):
            soup = BeautifulSoup(html_text, 'html.parser')
            # Remove script, style, and other irrelevant tags
            for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "input", "iframe", "img", "svg", "link", "meta", "noscript", "figure"]):
                element.extract()
            # Try finding common main content containers
            main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content') or soup.find(id='main-content') or soup.find(class_='main-content') or soup.find(role='main')
            target_element = main_content if main_content else soup.body
            if not target_element: return None # No body or main content found
            # Get text, trying to preserve some structure with newlines, then join lines
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
    if not api_key:
        logger.error("[Fallback API] URLToText API key was not provided for this call.")
        return None # Can't proceed without the key

    logger.info(f"[Fallback API] Attempting to fetch content for: {url} using URLToText API")
    api_endpoint = "https://urltotext.com/api/v1/urltotext/"
    payload = json.dumps({
        "url": url,
        "output_format": "text",
        "extract_main_content": True,
        "render_javascript": True, # Keep JS rendering enabled
        "residential_proxy": False,
        # "stealth_proxy": False, # Consider enabling if standard fails
        # "wait_for_js": 5000, # Consider increasing if needed
    })
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    try:
        logger.debug(f"[Fallback API] Sending request to URLToText API for {url}")
        # Run synchronous POST request in a thread
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
            except json.JSONDecodeError:
                logger.error(f"[Fallback API] Failed to decode JSON response from URLToText API for {url}. Response text: {response.text[:500]}...")
                return None
            except Exception as e:
                logger.error(f"[Fallback API] Error processing successful URLToText API response for {url}: {e}", exc_info=True)
                return None
        # Handle specific error codes
        elif response.status_code == 400: logger.error(f"[Fallback API] Bad Request (400) from URLToText API for {url}. Check parameters. Response: {response.text}"); return None
        elif response.status_code == 402: logger.error(f"[Fallback API] Payment Required (402) from URLToText API. Insufficient credits. URL: {url}"); return None
        elif response.status_code == 422: logger.error(f"[Fallback API] Invalid Request / Field Error (422) from URLToText API for {url}. Response: {response.text}"); return None
        elif response.status_code == 500: logger.error(f"[Fallback API] Internal Server Error (500) from URLToText API for {url}. Response: {response.text}"); return None
        else: logger.error(f"[Fallback API] Unexpected status code {response.status_code} from URLToText API for {url}. Response: {response.text}"); return None

    except requests.exceptions.Timeout: logger.error(f"[Fallback API] Timeout error connecting to URLToText API for {url}"); return None
    except requests.exceptions.RequestException as e: logger.error(f"[Fallback API] Request error connecting to URLToText API for {url}: {e}"); return None
    except Exception as e: logger.error(f"[Fallback API] Unexpected error during URLToText API call for {url}: {e}", exc_info=True); return None

# --- Gemini Summary Function ---
async def generate_summary(text: str, summary_type: str, api_key: str) -> str:
    """Generates summary using Gemini API asynchronously (running sync call in thread)."""
    logger.info(f"Generating {summary_type} summary. Input text length: {len(text)}")

    if not api_key:
         logger.error("Gemini API key was not provided to generate_summary.")
         return "Error: AI model configuration key is missing."

    try:
        logger.debug("Configuring Gemini client with provided key...")
        # Configuration should happen once ideally, but doing it here ensures key is set per call if needed
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash') # Using the specified model

        # Define prompts based on summary_type
        if summary_type == "paragraph":
            prompt = "You are an AI model designed to provide concise summaries using British English spellings. Your output MUST be: • Clear and simple language suitable for someone unfamiliar with the topic. • Uses British English spellings throughout. • Straightforward and understandable vocabulary; avoid complex terms. • Presented as ONE SINGLE PARAGRAPH. • No more than 85 words maximum. • Considers the entire text content equally. • Uses semicolons (;) instead of em dashes (– or —). Here is the text to summarise:"
        else: # points summary
            prompt = """You are an AI model designed to provide concise summaries using British English spellings. Your output MUST strictly follow this Markdown format:
• For each distinct topic or section identified in the text, create a heading.
• Each heading MUST be enclosed in double asterisks for bolding (e.g., **Section Title**).
• Immediately following each heading, list the key points as a bulleted list.
• Each bullet point MUST start with a hyphen and a space (`- `) on a new line.
• The text within each bullet point should NOT contain any bold formatting.
• Use clear, simple, and straightforward language suitable for someone unfamiliar with the topic.
• Use British English spellings throughout.
• Avoid overly complex or advanced vocabulary.
• Keep bullet points concise.
• Ensure the entire summary takes no more than two minutes to read.
• Consider the entire text's content, not just the beginning or a few topics.
• Use semicolons (;) instead of em dashes (– or —).

Here is the text to summarise:"""

        MAX_INPUT_LENGTH = 500000 # Keep truncation logic
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning(f"Input text length ({len(text)}) exceeds limit ({MAX_INPUT_LENGTH}). Truncating.")
            text = text[:MAX_INPUT_LENGTH] + "... (Content truncated)"
        full_prompt = f"{prompt}\n\n{text}"

        logger.debug("Sending request to Gemini model (via thread)...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # Run the synchronous generate_content call in a separate thread
        response = await asyncio.to_thread(
            model.generate_content,
            full_prompt,
            safety_settings=safety_settings
        )
        logger.debug("Received response from Gemini model.")

        summary = None
        try:
            if response.parts:
                 summary = "".join(part.text for part in response.parts)
            elif hasattr(response, 'text') and response.text:
                 summary = response.text # Fallback
        except ValueError as ve: logger.error(f"Could not extract text from Gemini response parts: {ve}. Response: {response}")
        except AttributeError as ae: logger.error(f"Unexpected response structure from Gemini: {ae}. Response: {response}")

        if summary:
            logger.info(f"Successfully generated summary. Output length: {len(summary)}")
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
        # Provide more specific user feedback based on error type if possible
        error_str = str(e).lower()
        if "api key not valid" in error_str or "permission denied" in error_str: return "Sorry, there was an issue connecting to the AI model (API key problem)."
        elif "model not found" in error_str: return "Sorry, the specified AI model is unavailable."
        elif "rate limit" in error_str: return "Sorry, the AI model is temporarily busy. Please try again in a moment."
        elif "deadline exceeded" in error_str or "timeout" in error_str: return "Sorry, the request to the AI model timed out. Please try again."
        return "Sorry, an unexpected error occurred while generating the summary with the AI model."


# --- Telegram Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"User {user.id} ({user.username or 'NoUsername'}) used /start.")
    mention = user.mention_html() if user.username else user.first_name
    await update.message.reply_html(f"👋 Hello {mention}! I can summarize YouTube links or website URLs.\n\nJust send me a link anytime!",)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🔍 **How to use this bot:**\n\n"
        "1. Send me any YouTube video link or website URL.\n"
        "2. I'll ask you how you want it summarized (paragraph or points).\n"
        "3. Click the button for your choice.\n"
        "4. Wait for the summary!\n\n"
        "If the first method fails for a website, I'll automatically try a secondary API (if configured).\n\n"
        "**Commands:**\n"
        "/start - Display welcome message\n"
        "/help - Show this help message",
        parse_mode='Markdown'
        )

async def handle_potential_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    url = update.message.text.strip()
    user = update.effective_user
    logger.info(f"User {user.id} sent potential URL: {url}")

    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')) or '.' not in url[8:]:
        logger.debug(f"Ignoring non-URL message from user {user.id}: {url}")
        return

    # Store URL in user context data before asking for summary type
    context.user_data['url_to_summarize'] = url
    logger.debug(f"Stored URL for user {user.id} in user_data")

    keyboard = [[InlineKeyboardButton("Paragraph Summary", callback_data="paragraph"), InlineKeyboardButton("Points Summary", callback_data="points")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Okay, I see this link:\n`{url}`\n\nHow would you like it summarized?", reply_markup=reply_markup, disable_web_page_preview=True, parse_mode='Markdown')

async def handle_summary_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query: return
    await query.answer() # Acknowledge button press

    summary_type = query.data
    user = update.effective_user or query.from_user # Get user info
    chat_id = query.message.chat_id # Get chat_id reliably

    logger.info(f"User {user.id} chose '{summary_type}' summary. Checking context for URL.")

    # Retrieve URL from context data, popping it to prevent reuse
    url = context.user_data.pop('url_to_summarize', None)

    if not url:
         logger.warning(f"User {user.id} pressed button, but NO URL found in user_data context (chat_id: {chat_id}).")
         try:
             await query.edit_message_text(text="Sorry, I couldn't find the URL associated with this request. Please send the link again.")
         except Exception as edit_err:
             logger.warning(f"Failed to edit message for missing URL: {edit_err}. Sending new message.")
             try: await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't find the URL associated with this request. Please send the link again.")
             except Exception as send_err: logger.error(f"Failed even to send fallback message for missing URL to chat {chat_id}: {send_err}")
         return # Cannot proceed without URL

    logger.debug(f"Retrieved URL '{url}' from user_data for user {user.id}")

    # --- Get API keys from environment variables ---
    current_gemini_key = os.environ.get(GEMINI_API_KEY_ENV)
    current_urltotext_key = os.environ.get(URLTOTEXT_API_KEY_ENV)

    # Check for essential Gemini key
    if not current_gemini_key:
        logger.error(f"CRITICAL: Environment variable {GEMINI_API_KEY_ENV} not found. Cannot generate summary.")
        await context.bot.send_message(chat_id=chat_id, text=f"Error: The AI model configuration ({GEMINI_API_KEY_ENV}) is missing on the server. Please contact the bot admin.")
        try: await query.delete_message() # Clean up button message
        except Exception: pass
        return # Cannot proceed without Gemini key

    # Check for optional URLToText key (only log if missing)
    if not current_urltotext_key:
        logger.warning(f"Environment variable {URLTOTEXT_API_KEY_ENV} not found. Website fallback scraping via API is disabled.")

    # --- Start Processing ---
    processing_message_text = f"Got it! Generating '{summary_type}' summary for:\n`{url}`\n\nThis might take a moment..."
    message_to_delete_later = None
    try:
        # Edit the original message (with buttons) to show processing status
        await query.edit_message_text(text=processing_message_text, parse_mode='Markdown', disable_web_page_preview=True)
    except Exception as e:
        logger.warning(f"Could not edit original message: {e}. Sending new status message.")
        try: # If editing fails (e.g., message too old), send a new one
            message_to_delete_later = await context.bot.send_message(chat_id=chat_id, text=processing_message_text, parse_mode='Markdown', disable_web_page_preview=True)
        except Exception as send_err:
            logger.error(f"Fatal: Failed to send status message to chat {chat_id}: {send_err}")
            return # Can't even notify user, stop processing

    content = None
    user_feedback_message = None # To store error messages for the user
    success = False # Track if we successfully generated a summary

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')

        is_youtube = is_youtube_url(url)

        if is_youtube:
            video_id = extract_youtube_id(url)
            if video_id:
                content = await get_youtube_transcript(video_id)
                if not content: user_feedback_message = "Sorry, I couldn't get the transcript for that YouTube video. It might be unavailable, private, or have transcripts disabled."
            else:
                user_feedback_message = "Sorry, I couldn't understand that YouTube URL format."
        else: # It's assumed to be a website URL
            # Try primary scraping method
            content = await get_website_content(url)
            if content:
                logger.info(f"Primary website scraping successful for {url}.")
            else:
                logger.warning(f"Primary scraping failed for {url}. Attempting fallback API if key exists.")
                # --- Fallback Logic ---
                if current_urltotext_key:
                    await context.bot.send_chat_action(chat_id=chat_id, action='typing') # Indicate more work
                    content = await get_website_content_via_api(url, current_urltotext_key)
                    if content:
                        logger.info(f"Fallback URLToText API scraping successful for {url}.")
                    else:
                        user_feedback_message = "Sorry, I couldn't fetch content from that website using either the primary or fallback method. The site might be inaccessible, block scraping, or be empty."
                        logger.error(f"Both primary and fallback API failed for {url}.")
                else:
                    # Primary failed, and no fallback key configured
                    user_feedback_message = "Sorry, I couldn't fetch content from that website using the primary method. The fallback method is not configured."
                    logger.warning(f"Primary scraping failed for {url}, and {URLTOTEXT_API_KEY_ENV} is missing.")
                # --- End Fallback Logic ---

        # --- Generate Summary if Content was Fetched ---
        if content:
            logger.info(f"Content fetched (length: {len(content)}), proceeding to generate summary.")
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            summary = await generate_summary(content, summary_type, current_gemini_key)

            # Check if summary generation itself returned an error/failure message
            if summary.startswith("Error:") or summary.startswith("Sorry,"):
                 user_feedback_message = summary # Use the error message from Gemini
                 logger.warning(f"Summary generation failed or returned error: {summary}")
            else:
                 # Send the successful summary
                 await context.bot.send_message(chat_id=chat_id, text=summary, parse_mode='Markdown', disable_web_page_preview=True)
                 success = True
                 user_feedback_message = None # Clear any previous failure message

        # --- Send Feedback if any step failed and no summary was sent ---
        if user_feedback_message and not success:
            await context.bot.send_message(chat_id=chat_id, text=user_feedback_message)

    except Exception as e:
        logger.error(f"Unexpected error during processing callback for URL {url} from user {user.id}: {e}", exc_info=True)
        try: # Try to inform the user about the unexpected error
            await context.bot.send_message(chat_id=chat_id, text="Oops! Something went wrong while processing your request. Please try again later or contact the admin if the problem persists.")
        except Exception as final_err:
             logger.error(f"Failed to send final error message to chat {chat_id}: {final_err}")
    finally:
        # Clean up the status message (either the edited one or the separately sent one)
        try:
             if message_to_delete_later: # If we sent a separate status message
                 await context.bot.delete_message(chat_id=chat_id, message_id=message_to_delete_later.message_id)
             elif query: # If we edited the original query message
                 # Decide whether to delete on success or failure
                 # Deleting is cleaner, but you could also remove the keyboard
                 await query.delete_message()
        except Exception as del_e:
             logger.warning(f"Could not delete status/button message for chat {chat_id}: {del_e}")


# --- Error handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)


# --- Main Function to Start the Bot ---
def main() -> None:
    """Runs the Telegram bot."""
    logger.info("Starting bot...")

    # --- Get TELEGRAM_TOKEN from Environment Variable ---
    token = os.environ.get(TELEGRAM_TOKEN_ENV)
    if not token:
        logger.critical(f"CRITICAL ERROR: Environment variable {TELEGRAM_TOKEN_ENV} not set.")
        sys.exit(f"Error: Bot token ({TELEGRAM_TOKEN_ENV}) not found in environment variables.")

    # --- Check for other required keys at startup (optional but good practice) ---
    if not os.environ.get(GEMINI_API_KEY_ENV):
         logger.warning(f"WARNING: Environment variable {GEMINI_API_KEY_ENV} not set. Summarization will fail.")
    else:
        logger.info(f"{GEMINI_API_KEY_ENV} found.")

    if not os.environ.get(URLTOTEXT_API_KEY_ENV):
         logger.warning(f"WARNING: Environment variable {URLTOTEXT_API_KEY_ENV} not set. Website fallback API will be unavailable.")
    else:
         logger.info(f"{URLTOTEXT_API_KEY_ENV} found.")


    # --- Create the Application ---
    application = Application.builder().token(token).build()

    # --- Register Handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_potential_url))
    application.add_handler(CallbackQueryHandler(handle_summary_type_callback))

    # --- Register Error Handler ---
    application.add_error_handler(error_handler)

    # --- Start Polling ---
    logger.info("Bot is configured. Starting polling...")
    # run_polling is blocking, it will run forever until interrupted
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot polling stopped.") # Will likely only be seen if manually stopped


if __name__ == '__main__':
    # Ensure the script runs the main function when executed directly
    main()
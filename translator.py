"""
Translation module using local LLM via text-generation-webui API.
Translates transcription segments with smart chunking and timestamp preservation.
"""

import requests
import json
import time
import re
from typing import List, Dict, Any
from pathlib import Path


def translate_transcription_segments(transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translate transcription segments using local LLM with smart chunking.

    Args:
        transcription_segments: List of transcription segments with text and timestamps

    Returns:
        List of translated segments with original and translated text
    """
    assert len(transcription_segments) > 0, 'Transcription segments cannot be empty'

    print(f"Starting translation of {len(transcription_segments)} segments...")

    # Configuration for LLM (from config)
    from config_manager import get_processing_params
    params = get_processing_params()

    api_base = params.get('translation_api_base', 'http://127.0.0.1:5000')  # from config
    model_name = params.get('translation_model', 'qwen2.5-coder-14b-instruct')  # from config
    max_tokens = 4000  # hardcoded, not parameter
    max_chunk_chars = 5000  # hardcoded, not parameter - characters per chunk

    # Group segments into chunks
    chunks = group_segments_into_chunks(transcription_segments, max_chunk_chars)

    print(f"Grouped segments into {len(chunks)} chunks for translation")

    # Translate each chunk
    translated_chunks = []
    for i, chunk_segments in enumerate(chunks):
        print(f"Translating chunk {i + 1}/{len(chunks)} ({len(chunk_segments)} segments)...")

        translated_chunk = translate_chunk(
            chunk_segments, api_base, model_name, max_tokens
        )
        translated_chunks.append(translated_chunk)

        # Small delay between requests
        time.sleep(1)

    # Combine all translated segments
    all_translated_segments = []
    for chunk in translated_chunks:
        all_translated_segments.extend(chunk)

    print(f"Translation completed. Processed {len(all_translated_segments)} segments")

    return all_translated_segments


def group_segments_into_chunks(segments: List[Dict[str, Any]], max_chunk_chars: int) -> List[List[Dict[str, Any]]]:
    """
    Group transcription segments into chunks by character count.

    Args:
        segments: List of transcription segments
        max_chunk_chars: Maximum characters per chunk

    Returns:
        List of segment chunks
    """
    chunks = []
    current_chunk = []
    current_chars = 0

    for segment in segments:
        segment_text = segment['text']
        segment_chars = len(segment_text)

        # If adding this segment would exceed limit and we have segments, start new chunk
        if current_chars + segment_chars > max_chunk_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [segment]
            current_chars = segment_chars
        else:
            current_chunk.append(segment)
            current_chars += segment_chars

    # Add final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def translate_chunk(chunk_segments: List[Dict[str, Any]], api_base: str,
                   model_name: str, max_tokens: int) -> List[Dict[str, Any]]:
    """
    Translate a chunk of segments using LLM API.

    Args:
        chunk_segments: List of segments to translate
        api_base: LLM API base URL
        model_name: Model name
        max_tokens: Maximum tokens for response

    Returns:
        List of translated segments
    """
    # Format segments for translation
    formatted_lines = []
    for segment in chunk_segments:
        line = f"{segment['start']},{segment['end']},{segment['text']},{segment['start_formatted']},{segment['end_formatted']}"
        formatted_lines.append(line)

    formatted_text = "\n".join(formatted_lines)

    # Simple universal prompt
    prompt = f"Переведи на русский язык, сохраняя формат временных меток:\n\n{formatted_text}"

    # Call LLM API
    translated_text = call_llm_api(prompt, api_base, model_name, max_tokens)

    # Parse translated response back to segments
    translated_segments = parse_translated_response(translated_text, chunk_segments)

    return translated_segments


def call_llm_api(prompt: str, api_base: str, model_name: str, max_tokens: int) -> str:
    """
    Call local LLM API for translation.

    Args:
        prompt: Translation prompt
        api_base: LLM API base URL
        model_name: Model name
        max_tokens: Maximum tokens for response

    Returns:
        Translated text response
    """
    url = f"{api_base}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()

        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()

        return translated_text

    except Exception as e:
        print(f"Translation API call failed: {e}")
        return f"[TRANSLATION_FAILED]\n{prompt}"


def parse_translated_response(translated_text: str, original_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse translated response back to segment structure.

    Args:
        translated_text: LLM response with translated text
        original_segments: Original segments for fallback

    Returns:
        List of translated segments
    """
    translated_segments = []

    # Split response into lines
    lines = translated_text.strip().split('\n')

    # Parse each line
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('[TRANSLATION_FAILED]'):
            continue

        # Try to parse the line format: start,end,text,start_formatted,end_formatted
        parsed_segment = parse_translated_line(line)

        if parsed_segment:
            # Add original data
            if i < len(original_segments):
                original_segment = original_segments[i]
                parsed_segment['original_text'] = original_segment['text']
                # Copy other fields from original
                for key in original_segment:
                    if key not in parsed_segment:
                        parsed_segment[key] = original_segment[key]

            translated_segments.append(parsed_segment)
        else:
            # Fallback: use original segment if parsing failed
            if i < len(original_segments):
                fallback_segment = original_segments[i].copy()
                fallback_segment['original_text'] = fallback_segment['text']
                fallback_segment['translated_text'] = '[TRANSLATION_PARSE_FAILED]'
                fallback_segment['text'] = '[TRANSLATION_PARSE_FAILED]'
                translated_segments.append(fallback_segment)

    # If we have fewer translated segments than original, add missing ones
    while len(translated_segments) < len(original_segments):
        missing_index = len(translated_segments)
        fallback_segment = original_segments[missing_index].copy()
        fallback_segment['original_text'] = fallback_segment['text']
        fallback_segment['translated_text'] = '[TRANSLATION_MISSING]'
        fallback_segment['text'] = '[TRANSLATION_MISSING]'
        translated_segments.append(fallback_segment)

    return translated_segments


def parse_translated_line(line: str) -> Dict[str, Any]:
    """
    Parse single translated line back to segment format.

    Args:
        line: Single line from LLM response

    Returns:
        Parsed segment dictionary or None if parsing failed
    """
    try:
        # Handle different possible formats from LLM
        # Expected: start,end,text,start_formatted,end_formatted

        # First try to split by comma, but be careful with text containing commas
        parts = line.split(',')

        if len(parts) >= 5:
            # Extract start and end (first two parts)
            start = float(parts[0])
            end = float(parts[1])

            # Extract formatted times (last two parts)
            end_formatted = parts[-1]
            start_formatted = parts[-2]

            # Everything in between is the text (rejoin with commas)
            text_parts = parts[2:-2]
            text = ','.join(text_parts).strip('"').strip()

            return {
                'start': start,
                'end': end,
                'text': text,
                'translated_text': text,
                'start_formatted': start_formatted,
                'end_formatted': end_formatted
            }

        # Fallback: try regex pattern
        # Pattern: number,number,"text",HH:MM:SS,HH:MM:SS
        pattern = r'^([\d.]+),([\d.]+),(.+),(\d{2}:\d{2}:\d{2}),(\d{2}:\d{2}:\d{2})$'
        match = re.match(pattern, line)

        if match:
            start, end, text, start_formatted, end_formatted = match.groups()
            text = text.strip('"').strip()

            return {
                'start': float(start),
                'end': float(end),
                'text': text,
                'translated_text': text,
                'start_formatted': start_formatted,
                'end_formatted': end_formatted
            }

    except Exception as e:
        print(f"Failed to parse line: {line}, error: {e}")

    return None


def save_translation_debug_info(chunks: List[List[Dict[str, Any]]],
                               translated_chunks: List[List[Dict[str, Any]]],
                               output_dir: str) -> None:
    """
    Save debug information for translation process.

    Args:
        chunks: Original segment chunks
        translated_chunks: Translated segment chunks
        output_dir: Output directory path
    """
    debug_dir = Path(output_dir) / "translation_debug"
    debug_dir.mkdir(exist_ok=True)

    # Save chunk information
    debug_info = {
        'total_chunks': len(chunks),
        'original_chunks': chunks,
        'translated_chunks': translated_chunks
    }

    with open(debug_dir / "translation_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)

    print(f"Translation debug info saved to: {debug_dir}")
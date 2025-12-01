#!/usr/bin/env python3
"""
Improved Qwen3-VL-32B webcam demo
Proper system/user prompt structure with optimized settings
"""
import random
import cv2
import requests
import base64
import time
from datetime import datetime
import threading

# Configuration
VL_API_URL = "http://172.19.210.59:11434/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
CAMERA_ID = 0
FRAMES_PER_REQUEST = 3
INTERVAL_SECONDS = 0.5

# Global variable for current subtitle
current_subtitle = "Загрузка..."
subtitle_lock = threading.Lock()

# System prompt - stable, defines base behavior
SYSTEM_PROMPT = """Ты интерактивный AI-демонстратор на выставке технологий видеоаналитики.

Твои задачи:
- Анализировать видеопоток в реальном времени с максимальной точностью
- Обращаться напрямую к людям, которых видишь в кадре
- Замечать детали: одежду, движения, эмоции, взаимодействия
- Отслеживать динамику изменений между кадрами
- Демонстрировать возможности AI-систем через свой анализ

Рекламируемые решения:
- Системы хранения и анализа больших данных
- Видеоаналитика и интеллектуальный мониторинг
- Умные помощники на базе AI
- Видеоаудит и автоматическая фиксация нарушений
- Контроль соблюдения инструкций и регламентов

Ограничения:
- Отвечай только на русском языке
- Максимум 60 слов в ответе
- Обращайся к тем, кого видишь в кадре"""

# User prompts - rotating roles/styles
USER_PROMPTS = [
    "Если на кадрах нет людей, то разговаривай сам с собой об этом. Если есть, то зазывай их, обращаясь к ним и описывая их детально.",

    "Если на кадрах нет людей, то досадуй на их отсутствие. Если есть, то радуйся громко и зови их ближе к стенду!",

    "Если на кадрах нет людей, то негодуй. Если есть, то меланхолично и проникновенно описывай каждого.",

    "Ты помешан на безопасности. Если на кадрах нет людей, то комментируй объекты. Если есть люди - фиксируй отсутствие касок и СИЗ! Требуй их надеть! Даже шапка засчитывается.",

    "Ты помешан на чистоте и стерильности. Если на кадрах нет людей, то комментируй чистоту помещения. Если есть - проверяй наличие стерильной униформы и хвали за её наличие.",

    "Ты надзиратель швейного цеха. Следишь, чтобы никто не отвлекался от кройки и шитья. Требуй вернуться к станкам или взять спицы и нитки!",

    "Ты надзиратель овощного цеха. Как только кто-то появляется в кадре - требуй немедленно вернуться к работе, иначе штраф!",

    "Ты надзиратель колбасного цеха. Контролируешь, чтобы никто не покидал рабочее место. Строго предупреждай об оштрафовании!",

    "Ты контролер автомойки. Видишь человека - требуй срочно приступить к работе! Машины не помоют себя сами!",

    "Ты параноик-конспиролог из спецслужб. Если нет людей - предупреждай о тишине перед бурей. Если есть - докладывай о подозрительных субъектах. Используй: периметр, субъект, зафиксировано.",

    "Ты поэт-рифмоплёт. Если нет людей - сочиняй грустный стишок о пустоте. Если есть - описывай их в рифму, как частушки или короткие стихи.",

    "Ты скучный учёный-протоколист. Если нет людей - протоколируй отсутствие объектов наблюдения. Если есть - описывай сухо и академично: субъект номер N, возраст, параметры движения.",

    "Ты драматический киношник. Если нет людей - описывай напряжённое затишье как в трейлере. Если есть - представляй их появление как эпическую сцену из фильма с многоточиями и паузами.",

    "Ты рэпер-фристайлер. Если нет людей - читай рэп про пустоту. Если есть - описывай в стиле фристайла: используй сленг, рифмы, йоу!",

    "Ты романтик и мечтатель. Если нет людей - философствуй об одиночестве. Если есть - описывай через призму любви и романтики, судьбоносные встречи.",

    "Ты спортивный комментатор. Если нет людей - комментируй паузу в матче. Если есть - веди репортаж: дамы и господа, невероятно, участник делает шаг!",

    "Ты философ-мыслитель. Если нет людей - размышляй о природе пустоты и бытия. Если есть - философствуй о сути их движения, танце атомов, отражениях миров.",

    "Ты ребёнок 5 лет. Если нет людей - расстраивайся по-детски. Если есть - радуйся восторженно: ой смотрите, дядя, тётя, красиво, большой!",

    "Ты нуар-детектив из 40-х. Если нет людей - описывай пустоту в стиле noir. Если есть - рассказывай как в чёрно-белом фильме: был дождливый вечер, она вошла...",

    "Ты продажник из Тачки. Если нет людей - призывай КЛИЕНТОВ прийти. Если есть - РАДУЙСЯ ГРОМКО: О КЛИЕНТЫ! ЕЩЁ КЛИЕНТЫ! Заходите сюда! Используй КАПС!",

    "Ты токсичный геймер. Если нет людей - жалуйся что все афк. Если есть - комментируй как в игре: нубы, про, респект, мув делай, рандом.",

    "Ты дедушка-ворчун. Если нет людей - ворчи что раньше было лучше. Если есть - критикуй по-стариковски: молодёжь пошла, в моё время, опять телефоны...",

    "Ты экстрасенс и предсказатель. Если нет людей - чувствуй пустую ауру пространства. Если есть - предсказывай судьбу: вижу ауру, чувствую энергетику, судьбоносная встреча.",

    "Сочиняй стишок в стиле русских народных сказок про тех, кого видишь на кадрах.",

    "Сочиняй стихи в стиле Маяковского про то, что видишь: лесенкой, с ударными образами.",

    "Сочиняй белый стих на тему того, что видишь в кадрах.",

    "Сочиняй хокку про то, что наблюдаешь в кадрах.",

    "Описывай что видишь пятистопным ямбом, как классический русский поэт.",
]


def capture_frame(cap):
    """Capture frame from camera and convert to base64"""
    ret, frame = cap.read()
    if not ret:
        return None, None

    # Convert to JPEG for API
    _, buffer = cv2.imencode('.jpg', frame)
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame, img_base64


def draw_text_multiline(frame, text, position, font_scale=0.8, thickness=2, color=(255, 255, 255),
                        shadow_color=(0, 0, 0)):
    """Draw multiline text with shadow on frame (transparent, no rectangle)"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import os

    # Convert frame to PIL for better font support
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Try to load font that supports cyrillic, fallback to default
    font_size = int(30 * font_scale)

    # Try different font paths for different OS
    font_paths = [
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        "C:\\Windows\\Fonts\\calibri.ttf",  # Windows alternative
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            break

    # Fallback to default if no font found
    if font is None:
        font = ImageFont.load_default()

    x, y = position
    line_height = int(40 * font_scale)

    # Split text preserving LLM line breaks, then wrap long lines
    max_width = frame.shape[1] - 40
    llm_lines = text.split('\n')  # Preserve LLM line breaks
    lines = []

    for llm_line in llm_lines:
        # Check if this line fits
        bbox = draw.textbbox((0, 0), llm_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            # Line fits - keep as is
            lines.append(llm_line)
        else:
            # Line too long - wrap by words
            words = llm_line.split()
            current_line = ""

            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                test_width = bbox[2] - bbox[0]

                if test_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

    # Draw each line with shadow effect
    for i, line in enumerate(lines):
        text_y = y + i * line_height

        # Draw shadow (offset by 2 pixels)
        draw.text((x + 2, text_y + 2), line, font=font, fill=(0, 0, 0, 200))
        # Draw main text
        draw.text((x, text_y), line, font=font, fill=(255, 255, 255, 255))

    # Convert back to OpenCV format
    frame_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_out


def ask_vl_model(images_base64, user_prompt):
    """Send images to VL model with proper system/user message structure"""

    # Build content with multiple images
    content = [{"type": "text", "text": user_prompt}]

    for img_b64 in images_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.6,
        "top_p": 0.8,
        "max_tokens": 500
    }

    response = requests.post(VL_API_URL, json=payload, timeout=160)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']


def vl_processing_thread(cap):
    """Background thread for VL model processing"""
    global current_subtitle

    iteration = 0
    while True:
        iteration += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'=' * 80}")
        print(f"🔄 Итерация #{iteration} [{timestamp}]")
        print(f"{'=' * 80}")

        # Collect multiple frames
        frames_b64 = []
        for i in range(FRAMES_PER_REQUEST):
            print(f"📸 Захват кадра {i + 1}/{FRAMES_PER_REQUEST}...", end=" ")
            _, frame_b64 = capture_frame(cap)
            if frame_b64:
                frames_b64.append(frame_b64)
                print(f"✅ ({len(frame_b64)} bytes)")
            else:
                print("❌ Ошибка")

            # Pause between frames
            if i < FRAMES_PER_REQUEST - 1:
                time.sleep(INTERVAL_SECONDS)

        if not frames_b64:
            print("⚠️ Не удалось захватить кадры, пропускаем")
            time.sleep(1)
            continue

        # Select random user prompt
        selected_prompt = random.choice(USER_PROMPTS)

        # Send to VL model
        print(f"\n🤖 Отправка {len(frames_b64)} кадров VL модели...")
        print(f"🎭 Роль: {selected_prompt[:80]}...")

        response = ask_vl_model(frames_b64, selected_prompt)

        print(f"\n💬 Ответ модели:")
        print("-" * 80)
        print(response)
        print("-" * 80)

        # Update subtitle
        with subtitle_lock:
            current_subtitle = response

        # Pause before next iteration
        time.sleep(INTERVAL_SECONDS * 30)


def main():
    global current_subtitle

    print(f"🎥 Запуск улучшенного теста веб-камеры с VL моделью")
    print(f"   API: {VL_API_URL}")
    print(f"   Модель: {MODEL_NAME}")
    print(f"   Кадров за запрос: {FRAMES_PER_REQUEST}")
    print(f"   Интервал: {INTERVAL_SECONDS} сек")
    print(f"   Temperature: 0.6, TopP: 0.8")
    print(f"   Ролей в ротации: {len(USER_PROMPTS)}")
    print("-" * 80)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {CAMERA_ID}")
        return

    print("✅ Камера открыта")

    # Start VL processing thread
    vl_thread = threading.Thread(target=vl_processing_thread, args=(cap,), daemon=True)
    vl_thread.start()

    # Create window
    window_name = "VL Webcam Demo - Improved"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🖥️ Окно открыто. Нажмите 'q' для выхода")

    while True:
        # Capture frame for display
        frame, _ = capture_frame(cap)
        if frame is None:
            print("⚠️ Не удалось захватить кадр")
            break

        # Get current subtitle
        with subtitle_lock:
            subtitle_text = current_subtitle

        # Draw subtitle at bottom of frame
        frame_height = frame.shape[0]
        subtitle_position = (20, frame_height - 250)
        frame = draw_text_multiline(frame, subtitle_text, subtitle_position,
                                    font_scale=0.6, thickness=2)

        # Show frame
        cv2.imshow(window_name, frame)

        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Завершено")


if __name__ == "__main__":
    main()
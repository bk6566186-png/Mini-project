from openai import OpenAI
import os
import datetime

# ===============================
# 🔐 API KEY SETUP
# ===============================
client = OpenAI(api_key="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")

# ===============================
# 🧠 CONTENT GENERATION FUNCTION
# ===============================
def generate_content(topic, platform):

    prompt = f"""
    Create engaging {platform} content in Hindi about: {topic}

    Include:
    - Attractive Hook Line
    - Short Main Content
    - 5 Trending Hashtags
    - Emojis
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional social media content creator."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ===============================
# 💾 SAVE FILE FUNCTION
# ===============================
def save_content(content, platform):

    # Create file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{platform}_{timestamp}.txt"

    # Folder path
    folder_path = f"content_storage/{platform.lower()}"

    # Create folder if not exists
    os.makedirs(folder_path, exist_ok=True)

    # Full file path
    file_path = os.path.join(folder_path, filename)

    # Write content into file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    return file_path


# ===============================
# 🚀 MAIN PROGRAM
# ===============================
if __name__ == "__main__":

    topic = input("Enter Topic: ")
    platform = input("Platform (instagram/whatsapp/facebook): ")

    print("\nGenerating Content...\n")

    content = generate_content(topic, platform)

    print("\n🔥 Generated Content:\n")
    print(content)

    saved_path = save_content(content, platform)

    print(f"\n✅ Content saved at: {saved_path}")

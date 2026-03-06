from flask import Flask, render_template_string, request

app = Flask(__name__)

# HTML Template (Bootstrap + Heart Vibes)
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Love Calculator 💘</title>
    <style>
        body { 
            background: linear-gradient(135deg, #ff758c, #ff7eb3);
            font-family: 'Arial', sans-serif; 
            text-align: center; 
            padding: 50px; 
            color: white;
        }
        .container {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            width: 400px;
            margin: auto;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        h1 { font-size: 2.5rem; margin-bottom: 20px; }
        input {
            padding: 10px; width: 80%; border: none;
            border-radius: 10px; margin: 10px;
            font-size: 1rem;
        }
        button {
            background: #ff1744; color: white; border: none;
            padding: 10px 20px; border-radius: 10px;
            font-size: 1rem; cursor: pointer;
        }
        button:hover { background: #d50000; }
        .result {
            margin-top: 20px;
            font-size: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💘 Love Calculator 💘</h1>
        <form method="POST">
            <input type="text" name="name1" placeholder="Your Name" required><br>
            <input type="text" name="name2" placeholder="Partner Name" required><br>
            <button type="submit">Calculate Love %</button>
        </form>

        {% if score %}
        <div class="result">
            <p><b>{{name1}}</b> ❤️ <b>{{name2}}</b> = <b>{{score}}%</b></p>
            <p>{{message}}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Love Messages
messages = [
    "💖 Perfect Match! Rab Ne Bana Di Jodi!",
    "😍 Wah! Dil garden garden ho gaya!",
    "😊 Acchi chemistry hai, aur mehnat karo!",
    "🔥 Tum dono ek dusre ke liye बने हो!",
    "🌹 Yeh pyaar amar rehne wala hai!",
    "😅 Pyaar zyada hai, bas thoda samajhna padega!",
    "💘 Tum dono ke beech spark hai!",
    "🥰 Aankhon hi aankhon mein ishara ho gaya!",
    "🌟 Love ka magic 100% confirm!",
    "🤗 Kabhi na chhodna ek dusre ka saath!"
]

# Logic Function
def calculate_love(name1, name2):
    # Thoda realistic logic (letters + ascii)
    combined = name1.lower() + name2.lower()
    score = (sum(ord(c) for c in combined) % 101)  # 0–100 %
    message = messages[score % len(messages)]
    return score, message

@app.route("/", methods=["GET", "POST"])
def home():
    score = None
    message = None
    name1 = ""
    name2 = ""
    if request.method == "POST":
        name1 = request.form["name1"]
        name2 = request.form["name2"]
        score, message = calculate_love(name1, name2)
    return render_template_string(html_template, score=score, message=message, name1=name1, name2=name2)

if __name__ == "__main__":
    app.run(debug=True)

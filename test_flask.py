
from flask import Flask

print("Starting Flask test...", flush=True)

app = Flask(__name__)

@app.route("/")
def home():
    return "Traffic Sign Recognition Flask test is working!"

if __name__ == "__main__":
    print("Opening server on port 5000...", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)

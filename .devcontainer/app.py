from flask import Flask

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return "Healthy", 200  # Responds with a 200 status if the service is healthy

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

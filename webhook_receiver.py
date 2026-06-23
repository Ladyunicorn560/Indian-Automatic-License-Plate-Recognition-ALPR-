from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()
    api_key = request.headers.get("x-api-key")
    print("\n" + "="*50)
    print("🚀 📥 RECEIVED NEW WEBHOOK FROM ALPR:")
    print(f"🔑 API Key: {api_key}")
    print("="*50)
    print(json.dumps(data, indent=4))
    print("="*50 + "\n")
    return {"status": "success", "message": "Webhook received successfully!"}

if __name__ == "__main__":
    import uvicorn
    print("\nStarting FAST-ALPR Webhook Receiver on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)

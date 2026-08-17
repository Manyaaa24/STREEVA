import asyncio
import websockets
import json

async def replay_trip(trip_id: str):
    uri = f"ws://localhost:8001/replay/{trip_id}"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Streaming trip data...\n")
            while True:
                message = await websocket.recv()
                
                if message == "Replay complete":
                    print("\n✅ Trip Replay Complete!")
                    break
                elif "not found" in message or "no points" in message:
                    print(f"❌ Error: {message}")
                    break
                    
                # Pretty print the JSON risk score
                data = json.loads(message)
                score = data['journey_risk_score']
                classification = data['classification']
                
                color_code = "\033[92m" # Green
                if classification == "High": color_code = "\033[93m" # Yellow
                elif classification == "Critical": color_code = "\033[91m" # Red
                    
                print(f"{color_code}[Live] Risk Score: {score:04.1f}/100 | {classification.ljust(8)} | "
                      f"Area: {data['fusion_factors']['area_risk']:04.1f} | "
                      f"Motion: {data['fusion_factors']['motion_anomaly']:04.1f}\033[0m")
                      
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    trip_id = input("Enter Trip ID to replay (e.g. trip_12345): ")
    asyncio.run(replay_trip(trip_id))

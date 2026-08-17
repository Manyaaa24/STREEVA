# Trip Data Collection Guide

This guide explains how to use the `logger.html` tool to collect real sensor data (GPS and Accelerometer) on your mobile device for the STREEVA Risk Fusion Engine.

## Setup Instructions

Because mobile browsers require a secure context to access the DeviceMotion (accelerometer) API, you have two options for loading `logger.html` on your phone:

### Option 1: Local HTTP Server with ngrok (Recommended)
This approach tunnels your local file to an HTTPS URL so your phone can access it seamlessly.

1. On your computer, open a terminal in the `fusion-service` folder.
2. Start a simple Python web server:
   ```bash
   python -m http.server 8080
   ```
3. In a new terminal window, start ngrok to expose port 8080:
   ```bash
   ngrok http 8080
   ```
4. Ngrok will output an `https://...` URL. Open this URL on your phone's browser.
5. Tap on `logger.html`.

### Option 2: Localhost (Advanced)
If you connect your phone to your computer via USB and use Chrome Port Forwarding, you can access the file via `http://localhost`, which browsers treat as a secure context.

## Collecting Trips

We need to collect 4 to 6 trips around campus to validate the anomaly model and risk fusion engine. For each trip, follow these steps:

1. **Open the App**: Load `logger.html` on your phone.
2. **Select Label**: Choose the appropriate scenario from the dropdown.
3. **Start**: Tap "Start Recording". You may be prompted to grant accelerometer/motion permissions. Accept them.
4. **Walk the Route**: Keep the phone in your hand or pocket just like you normally would.
5. **Stop & Save**: Tap "Stop Recording". 
6. **Export**: Tap "Export JSON" to download the trip file to your phone (you can transfer it to your computer later via AirDrop, Email, or Slack), OR if the Fusion API is running on your laptop and accessible via ngrok/local network, you can tap "Sync to Server".

### Required Scenarios

Please collect at least one of each scenario, each lasting about 2 to 5 minutes:

- **Normal Walk**: A standard, continuous walk along a primary path (e.g., from the library to a dorm). Keep a steady pace.
- **Route Deviation**: Start walking normally, then intentionally take a detour down a side lane or smaller path (this will test the area-risk GPS shift).
- **Stop-Test**: Walk for 30 seconds, then stand completely still for 45-60 seconds (simulating freezing or waiting), then resume walking.
- **Anomalous / Erratic**: Walk normally, then do something irregular—jog for a bit, quickly walk up/down a flight of stairs, or make sudden sharp orientation changes. (Note: Please do not actually fall and hurt yourself).

## Next Steps
Once you have collected the JSON files, place them in the `fusion-service/data/trips/` folder. The Risk Fusion Engine will use these to demonstrate real-time risk replay.

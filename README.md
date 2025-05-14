# Image processing Autonomous Docking


This repository contains the image processing and autonomous navigation system developed for the UiS Subsea's Yme ROV... 
Key feature: 

## 📚 Table of Contents
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Jetson Instructions (MCP2515 for Jetson Nano)](#jetson-instructions-mcp2515-for-jetson-nano)

- [How to Run the Code](#how-to-run-the-code) 

 3D Modeling: Structure from Motion (SfM) implementation for generating 3D models from 2D image sequences

   Autonomous Docking: Vision-based docking system using ArUco marker detection

   Pipeline Inspection: Color-based pipeline detection with elbow/bend recognition for autonomous tracking

   Visual Inspection: Detection of subsea structures and valve systems


## Prerequisites

- Python 3.7  
- GStreamer  
- FFmpeg  

---

## Jetson Instructions (MCP2515 for Jetson Nano)

Fixes problem on Jetson Nano while attempting to communicate via CAN bus driver (MCP2515). This repo is [fork from seeed](https://github.com/Seeed-Studio/seeed-linux-dtoverlays#readme).

> **NOTE:** It is recommended to reflash your microSD if you did modify the system. Just make sure everything working as intended.

## Pin Configuration

| MCP2515 | Jetson Nano |
| :------ | :---------- |
| VCC     | 5V          |
| GND     | GND         |
| CS      | 24          |
| MISO    | 21          |
| MOSI    | 19          |
| SCK     | 23          |
| INT     | 31          |

## How to Use

1. Open terminal app at your Jetson Nano
2. Let's clone this repo. To do that, run these commands
   ```
   git clone https://github.com/Thor-x86/seeed-linux-dtoverlays
   cd seeed-linux-dtoverlays
   ```
3a. Now replace the jetson-mcp2515.dts file with the one from subsea

   Github for subsea:
   https://github.com/UiS-Subsea/Kommunikasjon-2023

   filen ligger i JetsonNano_config mappen

   kopier filen og legg den i legg den i overlays/jetsonnano mappen til seeed-linux-dtoverlays-master repositoriet

   Husk å slette den gamle filen der.

3b. Then build and install with these commands
   ```
   make all_jetsonnano
   sudo make install_jetsonnano
   ```
   > **NOTE:** You'll asked for password to modify the system. It is normal when you type your password but nothing happens. Just type it as usual then hit enter.
4. After that, open utility to configure the Jetson
   ```
   sudo /opt/nvidia/jetson-io/jetson-io.py
   ```
   Dersom programmet åpner og lukker uten at noen valg kommer opp
   Da har en fil fra overlay mappen som ikke skal være der blitt kompilert og sendt til boot mappen i operativsystemet

   For å fikse det, så må man navigere til boot og slette den filen som ikke skal være der
   
5. Hit up or down arrow to navigate. Make sure ` Configure Jetson for compatible hardware` is highlighted then hit enter.
6. Navigate to `MCP251x CAN Controller` and hit enter to select
7. Choose `Save and reboot to reconfigure pins` and hit enter
8. Hit enter again to reboot

## Starting Up CAN Driver

1. Open the terminal again, then enter this command
   ```
   ip link
   ```
2. Make sure this word below shown at terminal
   ```
   can0: ...
   ```
   If not shown, possibly the wiring is not connected properly or too loose. Already correct? Then the fix is not working and you have to reflash the micro SD and start over.
3. Now let's configure the bitrate by entering this command
   ```
   sudo ip link set can0 type can bitrate <your-bitrate>
   ```
   Change `<your-bitrate>` to your intended bitrate **in bits unit**. As example `125000` for 125kbps. Bruk 500000
4. To start the CAN driver, enter this command
   ```
   sudo ifconfig can0 up
   ```
5. To stop the CAN driver, enter this command
   ```
   sudo ifconfig can0 down
   ```

## Testing the CAN Driver

- Read data from CAN bus
  ```
  candump can0
  ```
- Send "Hi!" text to CAN bus
  ```
  cansend can0 000#48.69.21.00
  ```


## How to Run the Code

This code is meant to be run through the [GUI](https://github.com/UiS-Subsea/GUI_2025).  
However, if you would like to run the individual files manually, follow the steps below:

1. **Clone the repository**

   ```bash
   git clone https://github.com/UiS-Subsea/Image-processing-Autonomous-Docking_2025.git
   cd Image-processing-Autonomous-Docking_2025



2. Python3 -m venv venv
   ```bash 
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies
   ```bash 
   pip install -r requirements.txt











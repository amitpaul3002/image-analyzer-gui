🕵️‍♂️ Image Forensics Tool
A GUI-based Python application for analyzing digital images to detect potential forgeries and assess image authenticity using metadata, error level analysis (ELA), and other forensic techniques.

📌 Features
🔍 Error Level Analysis (ELA) to detect possible tampering

🧬 Metadata Extraction (EXIF data)

🖼️ Noise Map & Histogram Analysis

⚙️ Easy-to-use PYQT GUI

💾 Supports JPEG, PNG, and other common formats

🚀 Getting Started
🔧 Prerequisites
Python 3.7+

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
You may also need:

bash
Copy
Edit
pip install opencv-python Pillow matplotlib piexif
▶️ Run the App
bash
Copy
Edit
python main.py
Replace main.py with your actual filename if different.

The tool applies forensic techniques such as:

ELA: Highlights altered regions by amplifying compression differences.

EXIF: Reveals details about image origin, editing, and camera settings.

Histogram: Shows pixel intensity distribution.

Noise Map: Detects inconsistencies in image regions.

📘 Usage Example
Launch the app

Load an image

Click buttons to perform:

Error Level Analysis

View Metadata

Show Histogram

Generate Noise Map

🛠️ Technologies Used
Python

PYQT (GUI)

OpenCV

Pillow

matplotlib

piexif



🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss.

📧 Contact
Created by [Amit Paul] – email: [2amit2pal@gmail.com]
Feel free to reach out!


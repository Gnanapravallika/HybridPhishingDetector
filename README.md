# HybridPhishingDetector
HybridPhishingDetector: A real-world phishing detection system combining ML, Deep Learning (LSTM), and threat intelligence (VirusTotal) to defend against adversarial URLs and improve generalization.

Project Highlights

✅ Combines traditional ML (Random Forest), Deep Learning (LSTM), and live VirusTotal threat scores  
✅ Uses real-world phishing dataset (PhishTank) and real benign samples  
✅ Improves on weaknesses of traditional models: evasion + low generalization  
✅ Evaluates with Accuracy, Precision, Recall, F1 + Visualizations  
✅ Includes loss comparison and hybrid prediction system  
✅ Final report included for presentation or resume

---

##  Motivation

Traditional phishing detectors often fail due to:
- Poor generalization on unseen domains
- Vulnerability to adversarial spoofed URLs

This project **hybridizes multiple data sources** (URL, HTML, VirusTotal), adds **deep learning**, and uses **real datasets** to create a more robust solution.

---

##  Tech Stack

- **Python**
- **Scikit-learn** (Random Forest)
- **PyTorch** (LSTM for adversarial URLs)
- **Pandas, BeautifulSoup, Requests**
- **VirusTotal API** (real-time threat lookup)
- **Matplotlib** (visualizations)

---

##  Project Structure

HybridPhishingDetector/
├── phishing_detector.py # Main Python logic
├── phishing_detection_notebook.ipynb # Jupyter version (optional)
├── requirements.txt # Python dependencies
├── .env # VirusTotal API key (user sets)
├── evaluation_metrics_plot.png # Accuracy, Precision, F1, Recall
├── loss_comparison_plot.png # ML vs DL vs Hybrid loss chart
├── Phishing_Detection_Final_Report.pdf # Formal project report
└── README.md # You're here

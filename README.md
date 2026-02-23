# 🛒 Retail Flow 2026
> **Precision Customer Analytics through Computer Vision**


Retail Flow is an intelligent computer vision system designed to transform physical movement into actionable retail intelligence. By utilizing **YOLO** machine learning models, the system tracks customer behavior, measures zone density, and correlates foot traffic with sales performance.

---

## 👥 The Team
**University:** Universidad Europea  
**Professor:** Alejandro Perdiguero  

| Name | Role | Position |
| :--- | :--- |:--- |
| **María Parra Ramos** | Software  Engineer | **CEO** |
| **Santiago Campos Snels** | Data Engineer|**Project Manager** |
| **Carlota Prósper Hernández** | Data Engineer | Worker |
| **Javier Hernández Pérez** | Data Scientist | Worker |
| **Vicente Vazquez** | Data Scientist | Worker |
| **Ashley Harris Velasquez** | Data Analyst | Worker |

---

## 🎯 Key Objectives
* 🔍 **Object Detection:** Real-time boundary box prediction to isolate human clients from store fixtures.
* 📈 **Traffic Analysis:** Granular tracking of user density and dwell time per zone.
* 💡 **Business Intelligence:** Bridging the gap between "interest" (browsing) and "conversion" (purchasing).

---

## 🏗️ Spatial Infrastructure
The system monitors a **100 ft x 50 ft** floor plan, strategically divided into five functional zones:

1.  **Entrance:** Control point for total conversion rate calculation.
2.  **Sales Floor:** Primary data source featuring high-engagement shelving units.
3.  **Fitting Rooms:** Lounge and trial area to analyze product attraction.
4.  **Self-Checkout:** Final transaction point for sales correlation.
5.  **Storage Room:** Restricted administrative zone.

---

## 📷 Technical Infrastructure (Camera Array)
We deploy a specialized 5-camera matrix to ensure zero blind spots and high-fidelity tracking:

| Hardware | Location | Field of View | Primary Goal |
| :--- | :--- | :--- | :--- |
| **Bullet Camera** | Entrance | 90° (30ft) | Entry/Exit Counting |
| **Fisheye Cam A** | Sales Floor | 360° (35ft) | Path Tracking & Heatmaps |
| **Fisheye Cam B** | Sales Floor | 360° (35ft) | Path Tracking & Heatmaps |
| **Dome Camera** | Fitting Rooms | 90° (30ft) | Dwell Time Measurement |
| **Turret Camera** | Checkout | 70° (15ft) | Transaction Correlation |

---

## 📊 Data Architecture
The project utilizes a robust relational SQL database structured to handle high-frequency detection data:

```mermaid
graph TD
    Store --> Zones
    Store --> Sales
    Zones --> Cameras
    Zones --> Traffic
    Cameras --> Detections
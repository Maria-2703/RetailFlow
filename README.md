# 🛒 Retail Flow 2026
> **Precision Customer Analytics through Computer Vision**


Retail Flow is an intelligent computer vision system designed to transform physical movement into actionable retail intelligence. By utilizing **YOLO (You Only Look Once)** machine learning models, the system tracks customer behavior, measures zone density, and correlates foot traffic with sales performance.

---

## 👥 The Team
**University:** Universidad Europea  
**Professor:** Alejandro Perdiguero  
**Date:** February 19, 2026  

| Name | Role |
| :--- | :--- |
| **Santiago Campos Snels** | Project Lead / Development |
| **Javier Hernández Pérez** | Systems Architecture |
| **María Parra Ramos** | Data Strategy |
| **Ashley Harris Velasquez** | ML Engineering |
| **Carlota Prósper Hernández** | Business Integration |

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
# Quick Start Guide - Enhanced UI Features

## 🎯 New Features at a Glance

### 1. Start/Stop Monitoring
**How to use:**
- Click `▶ Start Monitoring` or press `F5` to begin
- Click `⏹ Stop Monitoring` or press `F6` to stop
- Status indicator shows current state:
  - 🟢 **Running** (green) = Monitoring active
  - ⚫ **Stopped** (red) = Monitoring inactive

**Important Notes:**
- Start button is disabled when monitoring is active
- Stop button is disabled when monitoring is stopped
- This prevents accidental double-start/stop

---

### 2. Register Worker
**How to use:**
1. Click `➕ Register Worker` or press `Ctrl+N`
2. Enter Worker ID (or use auto-generated)
3. Enter Worker Name
4. Follow on-screen guidance to capture 7 face views:
   - Front (0°)
   - Slight left (15°)
   - Slight right (15°)
   - Medium left (30°)
   - Medium right (30°)
5. Click `Register Worker` when done

**Worker ID Replacement:**
- If you enter an existing Worker ID, the system will ask:
  ```
  Worker ID XXX already exists:
  
  Existing: John Doe
  New: Jane Smith
  
  Do you want to REPLACE the existing worker's photos?
  ```
- Click **Yes** to replace old photos with new ones
- Click **No** to cancel and use a different ID

---

### 3. View Worker Details
**How to use:**
1. Go to `Worker Status` tab
2. Select a worker from the table
3. Click `View Details` button

**Information shown:**
- Worker ID
- Name
- Position
- Contact
- Active status
- Registration date
- 2D Embeddings status (✓ or ✗)
- Number of photos stored
- Number of 3D views captured

---

### 4. Delete Worker
**How to use:**
1. Go to `Worker Status` tab
2. Select a worker from the table
3. Click `Delete Worker` button (red)
4. Confirm deletion in dialog

**What gets deleted:**
- ✓ Worker record
- ✓ All face photos
- ✓ 2D embeddings (ArcFace)
- ✓ 3D features (depth maps, mesh, landmarks)
- ✓ Activity logs
- ✓ Attendance records

**Warning:** This action CANNOT be undone!

---

## ⌨️ Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Start Monitoring | `F5` |
| Stop Monitoring | `F6` |
| Register New Worker | `Ctrl+N` |
| View Workers Tab | `Ctrl+W` |

---

## 📊 Worker Status Table

### Columns Explained

| Column | Description | Example |
|--------|-------------|---------|
| Worker ID | Unique identifier | 12345 |
| Name | Worker's full name | John Doe |
| Status | Current presence state | Present ✓ |
| Time Present | Duration on site | 15:30 |
| Last Seen | Last detection time | 14:25:10 |

### Status Colors

- 🟢 **Green (Present)** - Worker is currently on site
- 🔴 **Red (Exceeded)** - Worker has exceeded time limit
- ⚪ **Gray (Absent)** - Worker not currently detected

---

## 🔧 Common Tasks

### Updating Worker Photos
**Scenario:** Worker's appearance changed (haircut, glasses, etc.)

**Steps:**
1. Click `➕ Register Worker`
2. Enter the SAME Worker ID as before
3. Enter the name (can be same or updated)
4. Capture new photos
5. System asks to replace - click **Yes**
6. Old photos deleted, new photos saved ✓

### Removing a Worker
**Scenario:** Worker left company

**Steps:**
1. Go to `Worker Status` tab
2. Find and select the worker
3. Click `Delete Worker`
4. Confirm deletion
5. All data removed ✓

### Starting Daily Monitoring
**Steps:**
1. Launch application
2. Check camera feeds in `Cameras` tab
3. Press `F5` or click `▶ Start Monitoring`
4. Status indicator turns 🟢 green
5. Monitoring begins ✓

### Stopping at End of Day
**Steps:**
1. Press `F6` or click `⏹ Stop Monitoring`
2. Status indicator turns ⚫ red
3. Monitoring stops ✓
4. Generate reports if needed
5. Close application

---

## ⚠️ Important Notes

### Worker ID Rules
- Must be unique
- Range: 1 to 999,999
- Can replace existing ID (with confirmation)
- Auto-generated default: based on timestamp

### Face Capture Requirements
- **Minimum:** 3 views
- **Recommended:** 7 views (for best 3D coverage)
- **Views:** Front, ±15°, ±30°, ±45°
- **Quality:** Good lighting, face visible
- **Liveness:** System checks for real person

### Monitoring States
- **Running:** Cameras active, detection running
- **Stopped:** Cameras inactive, no detection
- **Transitions:** Smooth start/stop with visual feedback

### Data Persistence
- All worker data stored in PostgreSQL database
- Photos stored as binary (BYTEA)
- 3D features compressed and stored
- CASCADE delete ensures no orphaned data

---

## 🐛 Troubleshooting

### "Start button is grayed out"
**Solution:** Click Stop first, then Start again

### "Cannot delete worker"
**Solution:** Stop monitoring first, then delete

### "Worker ID already exists"
**Options:**
1. Click **Yes** to replace photos
2. Click **No** and use different ID

### "No camera detected"
**Check:**
1. Camera connected?
2. Camera permissions granted?
3. Check `config/config.json` for camera settings

### "Registration failed"
**Try:**
1. Ensure good lighting
2. Face clearly visible
3. Capture minimum 3 views
4. Check camera is working

---

## 💡 Tips & Best Practices

### For Best Recognition
1. Capture all 7 views during registration
2. Good lighting (avoid backlight)
3. Face 70% of frame
4. Follow on-screen guidance
5. Update photos if appearance changes

### For Efficient Management
1. Use meaningful Worker IDs (employee numbers)
2. Regularly review worker status
3. Delete inactive workers
4. Update photos every 6 months

### For Daily Operations
1. Start monitoring at shift beginning
2. Check status periodically
3. Review alerts tab
4. Stop monitoring at shift end
5. Generate daily reports

---

## 📱 UI Overview

```
┌─────────────────────────────────────────────────────────┐
│ Floor Monitoring System - Enhanced 3D Face Recognition  │
├─────────────────────────────────────────────────────────┤
│ ▶ Start │ ⏹ Stop │ ➕ Register │ 👥 Workers │ ⚙ Settings│
├─────────────────────────────────────────────────────────┤
│ ┌─ Cameras ─┬─ Worker Status ─┬─ Alerts ─┬─ Reports ─┐│
│ │           │                  │          │            ││
│ │  Camera   │  [Worker Table]  │ [Alerts] │ [Reports]  ││
│ │  Feeds    │                  │  List    │  View      ││
│ │           │  [View Details]  │          │            ││
│ │           │  [Delete Worker] │          │            ││
│ │           │  [Refresh]       │          │            ││
│ └───────────┴──────────────────┴──────────┴────────────┘│
├─────────────────────────────────────────────────────────┤
│ Ready - 3D Face Recognition System      🟢 Running      │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Training New Users

### Step-by-step First Use
1. **Launch application**
   - Double-click `run.py` or run from command line

2. **Register first worker**
   - Press `Ctrl+N` or click `➕ Register Worker`
   - Follow capture guidance
   - Complete registration

3. **Start monitoring**
   - Press `F5` or click `▶ Start Monitoring`
   - Check green status indicator

4. **View worker status**
   - Click `Worker Status` tab
   - See real-time updates

5. **Stop monitoring**
   - Press `F6` or click `⏹ Stop Monitoring`
   - Check red status indicator

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review `UI_ENHANCEMENTS.md` for technical details
3. Check application logs in `app.log`
4. Review PostgreSQL database logs

---

**Version:** 2.0 with 3D Face Recognition
**Last Updated:** November 2025
**System:** Enhanced UI with Worker Management

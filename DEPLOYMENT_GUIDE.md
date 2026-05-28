# 🚀 Render Deployment Guide - AeroSight AI

## Prerequisites
- GitHub account
- Render account (free tier available at [render.com](https://render.com))
- Git installed on your system

---

## 📋 Step-by-Step Deployment

### Step 1: Initialize Git Repository (if not already done)
```bash
git init
git add .
git commit -m "Initial commit - AeroSight AI"
```

### Step 2: Push to GitHub
```bash
# Create a new repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render

#### Option A: Using render.yaml (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and configure everything
5. Click **"Apply"** to deploy

#### Option B: Manual Setup
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `aerosight-ai` (or your choice)
   - **Region**: Singapore (or closest to you)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
   - **Plan**: Free
5. Click **"Create Web Service"**

### Step 4: Wait for Deployment
- First deployment takes 5-10 minutes
- Render will install dependencies and start your app
- You'll get a URL like: `https://aerosight-ai.onrender.com`

---

## ⚠️ Important Notes

### Model Files
Your model files (`*.keras`, `*.pt`) need to be in the repository:
- `transfer_learning_model.keras`
- `custom_cnn_model.keras`
- `yolov8n.pt`
- `runs/detect/aerial_yolo_model/weights/best.pt`

**If models are too large for GitHub (>100MB):**
1. Use Git LFS (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.keras"
   git lfs track "*.pt"
   git add .gitattributes
   git add .
   git commit -m "Add model files with LFS"
   git push
   ```

2. Or use external storage (Google Drive, Hugging Face) and download in app:
   ```python
   import gdown
   gdown.download('YOUR_GOOGLE_DRIVE_LINK', 'model.keras', quiet=False)
   ```

### Free Tier Limitations
- App sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds to wake up
- 750 hours/month free (enough for continuous running)

### Environment Variables (if needed)
Add in Render Dashboard → Environment:
```
PYTHON_VERSION=3.11.0
```

---

## 🔧 Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Verify all dependencies in `requirements.txt`
- Ensure Python version matches (3.11.0)

### App Crashes
- Check application logs in Render dashboard
- Verify model files are present
- Check memory usage (free tier has 512MB RAM limit)

### Slow Performance
- Free tier has limited resources
- Consider upgrading to paid plan for better performance
- Optimize model loading with caching

---

## 🎯 Quick Commands

```bash
# Check if git is initialized
git status

# Add all files
git add .

# Commit changes
git commit -m "Ready for deployment"

# Push to GitHub
git push origin main
```

---

## 📱 After Deployment

Your app will be live at: `https://your-app-name.onrender.com`

**Share your app:**
- Copy the URL from Render dashboard
- Test all features
- Monitor logs for any errors

---

## 🆘 Need Help?

- [Render Documentation](https://render.com/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy)
- Check Render logs for detailed error messages

---

**Developed by Ranjeet Kumar** 🚀

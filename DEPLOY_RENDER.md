# Deploying HausGPT to Render

This guide will help you deploy your HausGPT model to Render.

## Prerequisites

- GitHub account with your code pushed to a repository
- Hugging Face model published at: `jamelski/HausGPT`
- Render account (free tier works)

## Files Required for Deployment

The following files are already set up in this repository:

- `app.py` - Flask web application
- `templates/index.html` - Web interface
- `requirements.txt` - Python dependencies
- `Procfile` - Render deployment configuration

## Step-by-Step Deployment

### 1. Push Your Code to GitHub

Make sure all files are committed and pushed:

```bash
git add .
git commit -m "Add web interface for HausGPT"
git push origin main
```

### 2. Create a New Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** button
3. Select **"Web Service"**

### 3. Connect Your GitHub Repository

1. Click **"Connect GitHub"** (or use your existing connection)
2. Search for and select your repository: `jamelski/modelhaus`
3. Click **"Connect"**

### 4. Configure Your Web Service

Fill in the following settings:

**Basic Settings:**
- **Name**: `hausgpt` (or your preferred name)
- **Region**: Choose closest to you (e.g., `Oregon (US West)`)
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave blank (unless your code is in a subdirectory)

**Build & Deploy:**
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT`
  - (This is automatically read from `Procfile`, but you can specify it here too)

**Instance Type:**
- **Free** - Good for testing (512 MB RAM, will spin down after inactivity)
- **Starter** ($7/month) - 512 MB RAM, always on
- **Standard** ($25/month) - 2 GB RAM (recommended for better performance)

⚠️ **Note**: The model requires at least 500MB of RAM. The free tier works but may be slow.

### 5. Environment Variables (Optional)

If you need any environment variables, add them in the **Environment** section:

- Click **"Add Environment Variable"**
- Currently, no environment variables are required since the model ID is hardcoded

### 6. Deploy

1. Click **"Create Web Service"** at the bottom
2. Render will start building and deploying your app
3. You can watch the logs in real-time

### 7. Wait for Deployment

The deployment process includes:

1. **Building** - Installing dependencies (~2-3 minutes)
2. **Starting** - Loading the model (~1-2 minutes)
3. **Ready** - Your app is live!

You'll see your app URL at the top (e.g., `https://hausgpt.onrender.com`)

## What to Expect During First Launch

When your app first starts, you'll see in the logs:

```
Loading model and tokenizer...
Model loaded successfully on cpu
```

The first load may take 1-2 minutes as it downloads the model from Hugging Face.

## Accessing Your App

Once deployed, you can access your app at:
- **Your Render URL**: `https://YOUR-APP-NAME.onrender.com`

The interface will show:
- A text input for prompts
- Sliders for max_length, temperature, and top_p
- Example questions to try
- Generated responses from your model

## Troubleshooting

### App Won't Start
- Check the logs in Render dashboard
- Verify all files are committed and pushed
- Ensure `requirements.txt` is complete

### Out of Memory Errors
- Upgrade to a larger instance type (Standard recommended)
- Reduce the number of workers in the start command

### Model Loading Too Slow
- This is normal on the free tier
- Consider upgrading to Starter or Standard tier
- The model is cached after first load

### 502 Bad Gateway
- The app is still starting (model loading takes time)
- Wait 2-3 minutes after deployment starts
- Check logs for errors

## Testing Your Deployment

Once live, test with these example prompts:

1. "What is cyberspace according to JP 3-12?"
2. "What are the primary objectives of cyberspace operations?"
3. "Which organizations are responsible for cyberspace defense?"

## Monitoring

- **Logs**: View real-time logs in the Render dashboard
- **Metrics**: Monitor CPU and memory usage
- **Health Check**: Visit `/health` endpoint to check if the model is loaded

## Updating Your App

To update your deployed app:

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
   ```
3. Render will automatically rebuild and redeploy

## Cost Considerations

**Free Tier:**
- 750 hours/month free
- Spins down after 15 minutes of inactivity
- ~30 second cold start time
- Good for demos and testing

**Paid Tiers:**
- Always on (no spin down)
- Faster performance
- More memory for larger models
- Better for production use

## Advanced Configuration

### Custom Domain
1. Go to your service settings
2. Add a custom domain under **Settings > Custom Domains**
3. Follow DNS configuration instructions

### Scaling
- Increase worker count in start command (requires more memory)
- Upgrade instance type for more resources

### Environment Variables
If you want to make the model ID configurable:

1. Add environment variable: `MODEL_ID=jamelski/HausGPT`
2. Update `app.py` to read from environment:
   ```python
   MODEL_ID = os.environ.get('MODEL_ID', 'jamelski/HausGPT')
   ```

## Support

If you encounter issues:
1. Check the [Render documentation](https://render.com/docs)
2. Review your deployment logs
3. Test locally first: `python app.py`

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Visit http://localhost:5000
```

Good luck with your deployment!

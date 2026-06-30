# GitHub Upload Instructions

## Step 1: Create a New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `stock-index-pattern-prediction`
3. Description: `Desktop ML application for Hang Seng stock index forecasting using TPA-LSTM and Multivariate LSTM-FCN models`
4. Set to **Public**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **Create repository**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you a page with quick setup instructions. Copy the URL that looks like:
```
https://github.com/PavanKumar21-pv/stock-index-pattern-prediction.git
```

Then run these commands in your terminal (in the project directory):

```bash
# Add the remote repository
& 'C:\Program Files\Git\bin\git.exe' remote add origin https://github.com/PavanKumar21-pv/stock-index-pattern-prediction.git

# Verify the remote was added
& 'C:\Program Files\Git\bin\git.exe' remote -v

# Push to GitHub
& 'C:\Program Files\Git\bin\git.exe' push -u origin master
```

## Step 3: Authenticate with GitHub

When you run the push command, you may be prompted to authenticate:

### Option A: Using GitHub CLI (Recommended)
```bash
# Install GitHub CLI if you don't have it
winget install --id GitHub.cli

# Authenticate
gh auth login
```

### Option B: Using Personal Access Token
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. When prompted for password, use the token instead

### Option C: Using Git Credential Manager
```bash
# Install Git Credential Manager
& 'C:\Program Files\Git\bin\git.exe' config --global credential.helper manager

# Then push again - it will open a browser for authentication
& 'C:\Program Files\Git\bin\git.exe' push -u origin master
```

## Step 4: Take and Add Screenshots

After pushing to GitHub:

1. Run the application: `python Main.py`
2. Take screenshots of each feature using Windows Snipping Tool (Win + Shift + S)
3. Save screenshots in the `screenshots/` folder with these exact names:
   - `main-interface.png` - Main application window
   - `dataset-upload.png` - After uploading the dataset
   - `preprocessing.png` - After preprocessing
   - `svm-results.png` - SVM model results
   - `random-forest-results.png` - Random Forest results
   - `bayesian-ridge-results.png` - Bayesian Ridge results
   - `xgboost-results.png` - XGBoost results
   - `tpa-lstm-results.png` - TPA-LSTM results
   - `multivariate-lstm-fcn-results.png` - Multivariate LSTM-FCN results
   - `rmse-comparison.png` - RMSE comparison graph
   - `forecast-7-days.png` - 7-day forecast

4. Commit and push the screenshots:
```bash
& 'C:\Program Files\Git\bin\git.exe' add screenshots/*.png
& 'C:\Program Files\Git\bin\git.exe' commit -m "Add application screenshots"
& 'C:\Program Files\Git\bin\git.exe' push
```

## Step 5: Update README with Correct Repository Link

After successfully pushing, update the repository link in README.md:

1. Open README.md
2. Find the line: `https://github.com/pavankuma38767-bit/stock-index-pattern-prediction`
3. Replace with: `https://github.com/PavanKumar21-pv/stock-index-pattern-prediction`
4. Commit and push the change

## Quick Reference: All Git Commands

```bash
# Initialize (already done)
& 'C:\Program Files\Git\bin\git.exe' init

# Configure user (already done)
& 'C:\Program Files\Git\bin\git.exe' config user.name "PavanKumar21-pv"
& 'C:\Program Files\Git\bin\git.exe' config user.email "pavan@example.com"

# Add remote (replace with your actual repository URL)
& 'C:\Program Files\Git\bin\git.exe' remote add origin https://github.com/PavanKumar21-pv/stock-index-pattern-prediction.git

# Push to GitHub
& 'C:\Program Files\Git\bin\git.exe' push -u origin master

# For future updates
& 'C:\Program Files\Git\bin\git.exe' add .
& 'C:\Program Files\Git\bin\git.exe' commit -m "Your commit message"
& 'C:\Program Files\Git\bin\git.exe' push
```

## Troubleshooting

### If git command is not found:
Use the full path: `& 'C:\Program Files\Git\bin\git.exe' [command]`

### If you get authentication errors:
- Use GitHub CLI: `gh auth login`
- Or use a Personal Access Token as password
- Or install Git Credential Manager

### If remote already exists:
```bash
& 'C:\Program Files\Git\bin\git.exe' remote set-url origin https://github.com/PavanKumar21-pv/stock-index-pattern-prediction.git
```



# 🔐 Keystroke Dynamics Authentication System

A comprehensive web-based authentication system using keystroke dynamics and machine learning for user verification. This tool provides a complete solution for behavioral biometric authentication with an intuitive GUI.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [User Guide](#user-guide)
- [Admin Guide](#admin-guide)
- [Troubleshooting](#troubleshooting)
- [Model Requirements](#model-requirements)

## ✨ Features

### 🔑 User Management
- User registration with admin approval workflow
- Secure login/logout functionality
- Role-based access control (Admin and User roles)
- Session management

### 🤖 Model Integration
- Support for multiple ML models (Decision Tree, Random Forest, LightGBM)
- Dynamic model selection from models directory
- CSV test file upload capability
- Compatible with scikit-learn models

### 📊 Results & Reporting
- Comprehensive accuracy metrics (Accuracy, Precision, Recall, F1-Score)
- Interactive confusion matrix visualization
- Detailed classification reports
- Distribution analysis charts
- Export results (JSON and CSV formats)

### 📈 Dashboard
- **Admin Dashboard**: Complete system management
- **User Dashboard**: Model testing and activity tracking
- Real-time statistics and monitoring
- Modern, responsive UI with Plotly visualizations

### 👨‍💼 Admin Functions
- Approve/reject user registrations
- View all system users
- Monitor pending approvals
- Access model training interface
- System statistics overview

## 🔧 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 📦 Installation

### Step 1: Clone or Download the Project

Download all project files to your local directory.

### Step 2: Install Required Dependencies

Open your terminal/command prompt and run:

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.14.0
joblib>=1.2.0
```

## 📁 Project Structure

Organize your project directory as follows:

```
keystroke-auth-system/
│
├── app.py                          # Main application file
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── models/                         # Directory for trained models
│   ├── DecisionTree_model.pkl
│   ├── LightGBM_model.pkl
│   └── RandomForest_model.pkl
│
└── test_data/                      # Optional: Test datasets
    └── test_keystroke_data.csv
```

### Important Notes:
- Ensure the `models/` directory exists with your trained model files
- Model files must be in `.pkl` format
- Test data CSV files must include a `class` column (0 = authorized, 1 = unauthorized)

## 🚀 Running the Application

### Step 1: Navigate to Project Directory

```bash
cd path/to/keystroke-auth-system
```

### Step 2: Launch the Application

```bash
streamlit run app.py
```

### Step 3: Access the Application

The application will automatically open in your default web browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

## 👤 User Guide

### First Time Setup

1. **Access the Application**
   - Open your browser to `http://localhost:8501`

2. **Admin Login**
   - Username: `admin`
   - Password: `admin123`
   - Click "Login"

### For New Users

1. **Register an Account**
   - Click the "Register" tab
   - Enter desired username and password
   - Confirm password
   - Click "Register"
   - Wait for admin approval

2. **After Approval**
   - Login with your credentials
   - Access the User Dashboard

### Testing Models

1. **Select Model**
   - Choose models directory (default: `models`)
   - Select a model from the dropdown

2. **Upload Test Data**
   - Click "Browse files" or drag-and-drop
   - Select your CSV file with keystroke data
   - File must contain feature columns and a `class` column

3. **Run Evaluation**
   - Review data preview (optional)
   - Click "Run Evaluation"
   - Wait for processing

4. **View Results**
   - Check accuracy metrics
   - Review confusion matrix
   - Analyze classification report
   - Download results if needed

## 👨‍💼 Admin Guide

### User Management

1. **Approve/Reject Users**
   - Navigate to "User Management" tab
   - Review pending registrations
   - Click "Approve" or "Reject" for each user

2. **View All Users**
   - Scroll to "Approved Users" section
   - View user details and registration dates

### Model Training (Interface)

1. **Navigate to Training Tab**
   - Click "Training" tab in Admin Dashboard

2. **Upload Training Data**
   - Select CSV file with training data

3. **Configure Training**
   - Choose model type
   - Enter model name
   - Click "Start Training"

### System Monitoring

1. **View Statistics**
   - Check "System Stats" tab
   - Monitor total users, pending approvals
   - Review recent activity logs

## 🔧 Troubleshooting

### Issue: Model Loading Error (`invalid load key`)

**Solution 1:** Install joblib
```bash
pip install joblib
```

**Solution 2:** Re-save your models with compatible protocol
```python
import pickle
import joblib

# Load your model
model = joblib.load('models/YourModel.pkl')

# Re-save with compatible protocol
with open('models/YourModel_fixed.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)
```

### Issue: "No models found in directory"

**Solution:**
- Verify the `models/` directory exists
- Check that model files have `.pkl` extension
- Ensure correct path in "Models Directory" field

### Issue: CSV Upload Error

**Solution:**
- Verify CSV has a `class` column
- Check that feature columns match model training data
- Ensure CSV is properly formatted (comma-separated)
- Check for any special characters or encoding issues

### Issue: Port Already in Use

**Solution:**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

### Issue: Browser Not Opening Automatically

**Solution:**
- Manually open browser
- Navigate to the URL shown in terminal (usually `http://localhost:8501`)

## 📊 Model Requirements

### Expected Model Format
- **File Type:** `.pkl` (pickle format)
- **Compatibility:** scikit-learn compatible models
- **Supported Models:**
  - Decision Tree Classifier
  - Random Forest Classifier
  - LightGBM Classifier
  - Any scikit-learn compatible classifier

### Training Data Format
- CSV file with numerical features
- Feature columns: `feature_0`, `feature_1`, ..., `feature_n`
- Target column: `class` (0 = authorized, 1 = unauthorized)

### Example Dataset Structure

```csv
feature_0,feature_1,feature_2,...,feature_46,class
0.4089,-0.0469,-0.0741,...,0.3634,0
1.1010,4.3884,4.4301,...,0.3682,0
1.2831,-0.0192,0.0530,...,0.1781,0
```

## 🛡️ Security Notes

- Change default admin password after first login
- Use strong passwords for all accounts
- Regularly review user access
- Keep models and data secure
- Don't share credentials

## 📝 Default Credentials

**Admin Account:**
- Username: `admin`
- Password: `admin123`

⚠️ **Important:** Change this password immediately after first login!

## 💡 Tips

1. **For Best Performance:**
   - Use CSV files with consistent feature structure
   - Ensure models are trained on similar feature sets
   - Test with smaller datasets first

2. **For Multiple Models:**
   - Name models descriptively (e.g., `RandomForest_v2.pkl`)
   - Keep models organized in the `models/` directory
   - Document model versions and training dates

3. **For User Management:**
   - Regularly review pending approvals
   - Remove inactive users periodically
   - Monitor system activity logs

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check scikit-learn documentation: https://scikit-learn.org

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Author:** Keystroke Authentication System Team
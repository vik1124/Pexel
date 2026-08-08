# Start the local MLFlow tracking UI backed by SQLite.
# Open http://localhost:5000 in your browser after running this script.

$trackingUri = "sqlite:///mlflow.db"
$port        = 5000

Write-Host "Starting MLFlow UI on http://localhost:$port" -ForegroundColor Cyan
Write-Host "Backend: $trackingUri" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

mlflow ui --backend-store-uri $trackingUri --port $port

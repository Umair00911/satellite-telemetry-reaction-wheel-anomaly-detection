import http.server
import socketserver
import webbrowser
import os

PORT = 8080

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def start_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"🌐 Satellite Anomaly Detection System Dashboard running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == '__main__':
    start_server()

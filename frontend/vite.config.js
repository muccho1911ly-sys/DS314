import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backendProxy = {
  "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
  "/static": { target: "http://127.0.0.1:8000", changeOrigin: true },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  preview: { proxy: backendProxy },
  server: {
    proxy: backendProxy,
    host: "0.0.0.0",     // Lắng nghe trên toàn bộ giao diện mạng
    port: 5173,          // Cố định port 5173
    strictPort: true,    // Không tự động đổi port
    
    // Nếu "all" chạy tốt thì giữ nguyên, nếu lỗi hãy đổi thành mảng dưới đây:
    allowedHosts: [
      "app.tku.life", 
      ".tku.life"        // Cho phép tất cả subdomain của tku.life nếu cần
    ], 
  }
});

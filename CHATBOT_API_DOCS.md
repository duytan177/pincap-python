# Chatbot API Documentation

## Tổng quan

Chatbot sử dụng **Gemini Flash 2.5 + RAG** để quản lý và truy vấn media data. Chatbot chỉ sử dụng dữ liệu từ RAG, không tự tạo dữ liệu.

**Base URL:** `http://localhost:8000` (hoặc thay đổi theo môi trường của bạn)

**Endpoint:** `POST /api/v1/chatbot`

---


Dựa trên dữ liệu được cung cấp, dưới đây là các media phổ biến nhất:\n\n1.  **Tiêu đề:** (không có tiêu đề)\n    **Mô tả:** Đậu bắp xanh và những miếng gà nâu bóng được phục vụ trên đĩa đen hình chữ nhật. | Cận cảnh một món mì với thịt, đậu phộng và rau mùi, cùng với rau thơm tươi, tương ớt và bánh phồng tôm. | Hai bát súp mì châu Á với tôm, thịt heo, chả cá và hành phi trên bàn gỗ. | Một đùi gà chiên với trứng ốp la trên cơm vàng, ăn kèm cà chua thái lát, dưa chuột, dưa muối và sốt vàng trên đĩa đen. | Một túi nhựa trong suốt đựng Bánh Tráng Việt Nam, chứa chanh, nguyên liệu khô và chén nước sốt, có nhãn, trên bàn gỗ.\n2.  **Tiêu đề:** robot pink\n    **Mô tả:** Hai người trẻ, một nam và một nữ, đang đứng mỉm cười ngoài trời. Anh mặc áo khoác có khóa kéo màu đỏ sẫm; cô mặc đồng phục màu xanh với cổ áo trắng.\n3.  **Tiêu đề:** robot pink\n    **Mô tả:** Video mở đầu bằng một cảnh rộng của cánh đồng đầy hoa vàng rực rỡ, trải dài về phía bầu trời nhiều mây. Một phụ nữ trẻ mặc áo khoác đồng phục màu tối và một nam thanh niên mặc áo khoác thể thao màu đỏ sẫm đang đứng giữa những bông hoa. Họ quay mặt vào nhau, sau đó từ từ tiến lại gần, cuối cùng nắm tay. Cận cảnh cho thấy khuôn mặt người phụ nữ, ánh mắt cô hơi hướng xuống. Sau đó, cận cảnh người đàn ông cho thấy khuôn mặt anh với ánh mắt tương tự hướng xuống. Cảnh quay sau đó trở lại cặp đôi trong cánh đồng, và họ nghiêng người để trao nhau một nụ hôn.\n4.  **Tiêu đề:** robot pink\n    **M"


## Request Format

```json
{
  "user_id": "string (required)",
  "message": "string (required)",
  "conversation_history": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ],
  "suggested_media_ids": ["id1", "id2"],
  "file_url": "https://example.com/media.jpg"
}
```

### Parameters

- **user_id** (required): ID của người dùng
- **message** (required): Tin nhắn từ người dùng
- **conversation_history** (optional): Lịch sử hội thoại để context
- **suggested_media_ids** (optional): Danh sách media IDs khi tạo album
- **file_url** (optional): URL của file khi tạo media từ URL

---

## Response Format

```json
{
  "intent": "SEARCH_MEDIA|SUGGEST_MEDIA|CONFIRM_CREATE_ALBUM|CREATE_MEDIA_FROM_INPUT|GENERAL_QA",
  "answer": "string (optional)",
  "media": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "tags": ["tag1", "tag2"],
      "popularity_score": 0.95,
      "user_id": "string"
    }
  ],
  "ask_confirmation": {
    "action": "CREATE_ALBUM",
    "message": "string"
  },
  "action": "CREATE_ALBUM",
  "album": {
    "name": "string",
    "description": "string",
    "media_ids": ["id1", "id2"]
  },
  "frontend_link": "/media/create?prefill=true",
  "error": "string (optional)"
}
```

---

## Test Cases

### 1. SEARCH_MEDIA - Tìm kiếm và hỏi đáp về media

#### Case 1.1: Liệt kê media phổ biến

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Liệt kê 10 media phổ biến nhất hiện nay"
  }'
```

**Expected Response:**
```json
{
  "intent": "SEARCH_MEDIA",
  "answer": "Dưới đây là 10 media phổ biến nhất: ...",
  "media": [
    {
      "id": "media_001",
      "title": "Media Title",
      "description": "Media description",
      "tags": ["tag1", "tag2"],
      "popularity_score": 0.95,
      "user_id": "user456"
    }
  ]
}
```

#### Case 1.2: Tìm media theo chủ đề

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tìm media về anime One Piece"
  }'
```

#### Case 1.3: Tìm media với số lượng cụ thể

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Cho tôi xem 5 media về phong cảnh"
  }'
```

#### Case 1.4: Câu hỏi về media

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Media nào có tag anime?"
  }'
```

---

### 2. SUGGEST_MEDIA - Gợi ý media và xác nhận tạo album

#### Case 2.1: Gợi ý media theo chủ đề

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Gợi ý cho tôi 20 media chủ đề anime One Piece"
  }'
```

**Expected Response:**
```json
{
  "intent": "SUGGEST_MEDIA",
  "answer": "Tôi gợi ý các media sau về chủ đề One Piece: ...",
  "media": [
    {
      "id": "media_001",
      "title": "One Piece Episode 1",
      "description": "...",
      "tags": ["anime", "one-piece"],
      "popularity_score": 0.88,
      "user_id": "user456"
    }
  ],
  "ask_confirmation": {
    "action": "CREATE_ALBUM",
    "message": "Bạn có muốn tạo album từ các media này không?"
  }
}
```

#### Case 2.2: Gợi ý media với số lượng khác

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Gợi ý 15 media về phong cảnh thiên nhiên"
  }'
```

#### Case 2.3: Gợi ý media theo tag

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Đề xuất media có tag travel"
  }'
```

---

### 3. CONFIRM_CREATE_ALBUM - Xác nhận tạo album

#### Case 3.1: Xác nhận tạo album từ suggested media

**Bước 1:** Gợi ý media (như Case 2.1) để lấy `suggested_media_ids`

**Bước 2:** Xác nhận tạo album

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Có, tạo album đi",
    "suggested_media_ids": ["media_001", "media_002", "media_003"]
  }'
```

**Expected Response:**
```json
{
  "intent": "CONFIRM_CREATE_ALBUM",
  "action": "CREATE_ALBUM",
  "album": {
    "name": "One Piece Collection",
    "description": "Album chứa các media về anime One Piece",
    "media_ids": ["media_001", "media_002", "media_003"]
  }
}
```

#### Case 3.2: Xác nhận bằng tiếng Việt

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Đồng ý",
    "suggested_media_ids": ["media_001", "media_002"]
  }'
```

#### Case 3.3: Xác nhận bằng tiếng Anh

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "OK, tạo album",
    "suggested_media_ids": ["media_001", "media_002", "media_003", "media_004"]
  }'
```

#### Case 3.4: Từ chối (không tạo album)

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Không, cảm ơn"
  }'
```

**Expected Response:** Intent sẽ là `GENERAL_QA` hoặc không có action `CREATE_ALBUM`

---

### 4. CREATE_MEDIA_FROM_INPUT - Tạo media từ file/URL

#### Case 4.1: Tạo media từ URL

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tạo media từ URL này: https://example.com/image.jpg"
  }'
```

**Expected Response:**
```json
{
  "intent": "CREATE_MEDIA_FROM_INPUT",
  "media": {
    "title": "Generated Title",
    "description": "Generated description from AI analysis",
    "tags": ["tag1", "tag2", "tag3"],
    "media_url": "https://example.com/image.jpg"
  },
  "frontend_link": "/media/create?prefill=true"
}
```

#### Case 4.2: Tạo media với file_url parameter

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Thêm media từ file này",
    "file_url": "https://example.com/video.mp4"
  }'
```

#### Case 4.3: Tạo media từ nhiều URL trong message

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tạo media từ: https://example.com/image1.jpg và https://example.com/image2.png"
  }'
```

**Note:** Chỉ URL đầu tiên sẽ được xử lý.

---

### 5. GENERAL_QA - Câu hỏi chung

#### Case 5.1: Câu hỏi về hệ thống

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Chatbot này làm gì?"
  }'
```

**Expected Response:**
```json
{
  "intent": "GENERAL_QA",
  "answer": "Tôi là chatbot hỗ trợ quản lý media. Tôi có thể giúp bạn tìm kiếm, gợi ý media, tạo album và tạo media mới..."
}
```

#### Case 5.2: Câu hỏi về tính năng

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Làm thế nào để tìm media?"
  }'
```

#### Case 5.3: Chào hỏi

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Xin chào"
  }'
```

---

## Conversation Flow - Ví dụ đầy đủ

### Flow 1: Tìm kiếm → Gợi ý → Tạo Album

**Step 1: Tìm kiếm media**
```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tìm 5 media về anime"
  }'
```

**Step 2: Gợi ý media (lưu media_ids từ response)**
```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Gợi ý 10 media về One Piece"
  }'
```

**Response có `ask_confirmation` và danh sách `media` với các `id`**

**Step 3: Xác nhận tạo album (dùng media_ids từ Step 2)**
```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Có, tạo album",
    "suggested_media_ids": ["media_001", "media_002", "media_003", "media_004", "media_005"]
  }'
```

---

### Flow 2: Tạo media từ URL

**Step 1: Tạo media**
```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tạo media từ https://example.com/beautiful-sunset.jpg"
  }'
```

**Response có `media` object với title, description, tags và `frontend_link`**

---

## Conversation History - Sử dụng context

### Ví dụ với conversation history

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Còn media nào khác không?",
    "conversation_history": [
      {
        "role": "user",
        "content": "Tìm 5 media về anime"
      },
      {
        "role": "assistant",
        "content": "Tôi tìm thấy 5 media về anime: ..."
      }
    ]
  }'
```

---

## Error Handling

### Case: Không tìm thấy media

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tìm media về chủ đề không tồn tại xyz123"
  }'
```

**Expected Response:**
```json
{
  "intent": "SEARCH_MEDIA",
  "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp với yêu cầu của bạn.",
  "media": []
}
```

### Case: Tạo album không có media_ids

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tạo album",
    "suggested_media_ids": null
  }'
```

**Expected Response:**
```json
{
  "intent": "CONFIRM_CREATE_ALBUM",
  "error": "Không có danh sách media để tạo album. Vui lòng yêu cầu gợi ý media trước."
}
```

### Case: URL không hợp lệ

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Tạo media từ URL không hợp lệ"
  }'
```

**Expected Response:**
```json
{
  "intent": "CREATE_MEDIA_FROM_INPUT",
  "error": "Không tìm thấy URL hoặc file để tạo media."
}
```

---

## Testing với Postman

### Import Collection

1. Tạo collection mới trong Postman
2. Thêm request với method `POST`
3. URL: `http://localhost:8000/api/v1/chatbot`
4. Headers: `Content-Type: application/json`
5. Body (raw JSON): Sử dụng các ví dụ JSON ở trên

### Environment Variables

Tạo environment variables:
- `base_url`: `http://localhost:8000`
- `user_id`: `user123`

Sử dụng: `{{base_url}}/api/v1/chatbot`

---

## Notes

1. **RAG Context Only**: Chatbot chỉ sử dụng dữ liệu từ Elasticsearch, không tự tạo dữ liệu
2. **Vietnamese Responses**: Tất cả câu trả lời đều bằng tiếng Việt
3. **Intent Detection**: Tự động phát hiện intent từ message
4. **Media Filtering**: Tự động lọc media từ users bị block
5. **Embedding Search**: Sử dụng vector similarity search với min_score threshold
6. **Album Creation**: Chỉ trả về payload, không persist trực tiếp

---

## Quick Test Script

Tạo file `test_chatbot.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"
USER_ID="user123"

echo "=== Test 1: SEARCH_MEDIA ==="
curl -X POST "${BASE_URL}/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"${USER_ID}\", \"message\": \"Liệt kê 10 media phổ biến nhất\"}"

echo -e "\n\n=== Test 2: SUGGEST_MEDIA ==="
curl -X POST "${BASE_URL}/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"${USER_ID}\", \"message\": \"Gợi ý 20 media chủ đề anime One Piece\"}"

echo -e "\n\n=== Test 3: GENERAL_QA ==="
curl -X POST "${BASE_URL}/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"${USER_ID}\", \"message\": \"Chatbot này làm gì?\"}"

echo -e "\n\n=== Test 4: CREATE_MEDIA_FROM_INPUT ==="
curl -X POST "${BASE_URL}/api/v1/chatbot" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"${USER_ID}\", \"message\": \"Tạo media từ URL này: https://example.com/image.jpg\"}"
```

Chạy: `chmod +x test_chatbot.sh && ./test_chatbot.sh`

---

## API Response Examples

### Full Response - SEARCH_MEDIA
```json
{
  "intent": "SEARCH_MEDIA",
  "answer": "Dưới đây là 10 media phổ biến nhất hiện nay:\n\n1. Media Title 1 - Mô tả ngắn...\n2. Media Title 2 - Mô tả ngắn...",
  "media": [
    {
      "id": "media_001",
      "title": "Media Title 1",
      "description": "Full description of media 1",
      "tags": ["tag1", "tag2"],
      "popularity_score": 0.95,
      "user_id": "user456"
    },
    {
      "id": "media_002",
      "title": "Media Title 2",
      "description": "Full description of media 2",
      "tags": ["tag3", "tag4"],
      "popularity_score": 0.92,
      "user_id": "user789"
    }
  ]
}
```

### Full Response - SUGGEST_MEDIA
```json
{
  "intent": "SUGGEST_MEDIA",
  "answer": "Tôi gợi ý các media sau về chủ đề One Piece:\n\n1. One Piece Episode 1...\n2. One Piece Episode 2...",
  "media": [
    {
      "id": "media_001",
      "title": "One Piece Episode 1",
      "description": "Episode đầu tiên của One Piece",
      "tags": ["anime", "one-piece"],
      "popularity_score": 0.88,
      "user_id": "user456"
    }
  ],
  "ask_confirmation": {
    "action": "CREATE_ALBUM",
    "message": "Bạn có muốn tạo album từ các media này không?"
  }
}
```

### Full Response - CONFIRM_CREATE_ALBUM
```json
{
  "intent": "CONFIRM_CREATE_ALBUM",
  "action": "CREATE_ALBUM",
  "album": {
    "name": "One Piece Collection",
    "description": "Album chứa các media về anime One Piece, bao gồm các episode và hình ảnh liên quan.",
    "media_ids": ["media_001", "media_002", "media_003", "media_004", "media_005"]
  }
}
```

### Full Response - CREATE_MEDIA_FROM_INPUT
```json
{
  "intent": "CREATE_MEDIA_FROM_INPUT",
  "media": {
    "title": "Beautiful Sunset",
    "description": "A stunning sunset over the ocean with vibrant colors",
    "tags": ["sunset", "ocean", "nature", "landscape"],
    "media_url": "https://example.com/image.jpg"
  },
  "frontend_link": "/media/create?prefill=true"
}
```

---

## Troubleshooting

### Lỗi: Connection refused
- Kiểm tra server có đang chạy không: `curl http://localhost:8000/`
- Kiểm tra port 8000 có bị chiếm không

### Lỗi: No media found
- Kiểm tra Elasticsearch có dữ liệu không
- Kiểm tra user_id có bị block không
- Giảm `min_score` trong code nếu cần

### Lỗi: Gemini API error
- Kiểm tra `GEMINI_API_KEY` trong environment
- Kiểm tra quota của Gemini API

---

**Last Updated:** 2024


# Install
'''
git clone https://github.com/r4ndompuff/HW-llama.git
cd HW-llama
docker build -t llm_username:v1 .
docker run -p 8080:8080 llm_username:v1
'''

# Use
'''
curl -X 'POST' \
  'http://0.0.0.0:8080/message' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "Привет, расскажи мне про кредиты?",
  "user_id": "1234"
}'
'''

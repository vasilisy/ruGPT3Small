from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import HTTPException, Form
from fastapi.templating import Jinja2Templates

# Создание экземпляра FastAPI
app = FastAPI()

# Загрузка предобученной модели GPT-3 и токенизатора
model_name = 'ai-forever/rugpt3small_based_on_gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Инициализация шаблонов Jinja2 из папки templates
templates = Jinja2Templates(directory="templates")


# Обработчик POST-запроса для генерации текста
@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, user_input: str = Form(...)):
    # Токенизация введенного пользователем текста
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Генерация текста с использованием модели
    output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Декодирование сгенерированного текста из числовых идентификаторов
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Возврат HTML-страницы с результатами
    return templates.TemplateResponse("index.html", {"request": request, "user_input": user_input, "generated_text": generated_text})


# Обработчик GET-запроса для отображения домашней страницы
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Обработчик исключений HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


# Запуск приложения FastAPI
if __name__ == '__main__':
    import uvicorn

    # Запуск сервера с указанными параметрами
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)

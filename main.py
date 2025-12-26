import os, uuid, shutil, cv2, gc, logging
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- CALLING YOUR SPECIFIC FUNCTIONS ---
from layer1_tracking import run_layer_1
from layer2_analysis import analyze_match_broadcast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ELITESHOT")

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

# Auto-Setup Directories
for d in ["static", "templates", "projects"]: 
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
progress_db = {}

def full_tactical_pipeline(pid, filename):
    try:
        p_path = os.path.join(PROJECTS_DIR, pid)
        vid_in = os.path.join(p_path, filename)
        vid_out = os.path.join(p_path, "final_output.mp4")
        json_out = os.path.join(p_path, "match_data.json")
        stub_path = os.path.join(p_path, "track_stub.pkl")

        # --- EXECUTE YOUR LAYER 1 ---
        progress_db[pid] = {"percent": 10, "stage": "AI_TRACKING"}
        logger.info(f"[{pid}] EXECUTING LAYER 1...")
        run_layer_1(vid_in, stub_path) #
        
        gc.collect() 

        # --- EXECUTE YOUR LAYER 2 ---
        progress_db[pid].update({"percent": 50, "stage": "TACTICAL_ANALYSIS"})
        logger.info(f"[{pid}] EXECUTING LAYER 2...")
        analyze_match_broadcast(vid_in, stub_path, json_out, vid_out) #

        progress_db[pid] = {"percent": 100, "stage": "COMPLETE"}
        logger.info(f"[{pid}] SUCCESS: Project {pid} finished.")

    except Exception as e:
        logger.error(f"[{pid}] FATAL ERROR: {e}")
        progress_db[pid] = {"percent": -1, "stage": "ERROR"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status/{pid}")
async def get_status(pid: str):
    return {"data": progress_db.get(pid, {"percent": 0, "stage": "QUEUED"})}

@app.post("/upload")
async def upload(bg: BackgroundTasks, files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        pid = str(uuid.uuid4())[:8]
        p_path = os.path.join(PROJECTS_DIR, pid)
        os.makedirs(p_path, exist_ok=True)
        local_file = os.path.join(p_path, f.filename)
        with open(local_file, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        bg.add_task(full_tactical_pipeline, pid, f.filename)
        results.append({"id": pid, "filename": f.filename})
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
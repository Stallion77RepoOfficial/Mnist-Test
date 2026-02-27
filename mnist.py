import os
import pygame
import numpy as np
import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_saved_models():
    models = []
    for f in os.listdir():
        if f.startswith("mnist_weights_") and f.endswith(".npz"):
            parts = f.replace("mnist_weights_", "").replace(".npz", "").split("_")
            if len(parts) == 3:
                models.append((int(parts[0]), int(parts[1]), int(parts[2]), f))
    return sorted(models)

def load_or_train_model(h1, h2, max_iter, force=False):
    weight_file = f"mnist_weights_{h1}_{h2}_{max_iter}.npz"
    if not force and os.path.exists(weight_file):
        try:
            data = np.load(weight_file, allow_pickle=True)
            return data['weights'], data['biases']
        except Exception:
            os.remove(weight_file)
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = X / 255.0
    
    mlp = MLPClassifier(hidden_layer_sizes=(h1, h2), activation='logistic', max_iter=max_iter, random_state=42)
    mlp.fit(X[:35000], y[:35000])
    
    weights_obj = np.array(mlp.coefs_, dtype=object)
    biases_obj = np.array(mlp.intercepts_, dtype=object)
    
    np.savez(weight_file, weights=weights_obj, biases=biases_obj)
    return weights_obj, biases_obj

def center_image(grid):
    if np.sum(grid) == 0: return grid
    y_coords, x_coords = np.nonzero(grid)
    y_min, y_max, x_min, x_max = np.min(y_coords), np.max(y_coords), np.min(x_coords), np.max(x_coords)
    cropped = grid[y_min:y_max+1, x_min:x_max+1]
    h, w = cropped.shape
    centered = np.zeros((28, 28))
    centered[(28-h)//2 : (28-h)//2+h, (28-w)//2 : (28-w)//2+w] = cropped
    return centered

class NeuralNetwork:
    def __init__(self, h1=32, h2=32, iters=100, force=False):
        self.weights, self.biases = load_or_train_model(h1, h2, iters, force)
        self.activations = []

    def sigmoid(self, z): return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    def softmax(self, z):
        ex = np.exp(z - np.max(z))
        return ex / ex.sum()

    def forward(self, input_data):
        self.activations = [input_data.flatten()]
        curr = self.activations[0]
        for i in range(len(self.weights)):
            z = np.dot(curr, self.weights[i]) + self.biases[i]
            curr = self.softmax(z) if i == len(self.weights)-1 else self.sigmoid(z)
            self.activations.append(curr)
        return curr

class App:
    def __init__(self):
        pygame.init()
        self.info = pygame.display.Info()
        self.w, self.h = self.info.current_w, self.info.current_h
        self.screen = pygame.display.set_mode((self.w, self.h), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.font_xs = pygame.font.SysFont("Courier New", 10, bold=True)
        self.font_sm = pygame.font.SysFont("Courier New", 14, bold=True)
        self.font_md = pygame.font.SysFont("Courier New", 18, bold=True)
        self.font_lg = pygame.font.SysFont("Courier New", 28, bold=True)
        
        self.h1_val, self.h2_val, self.iter_val = 32, 32, 100
        self.is_training = self.is_loading = self.show_settings = self.show_info = False
        self.saved_models = get_saved_models()
        self.selected_model_idx = 0

        self.grid = np.zeros((28, 28))
        self.nn = NeuralNetwork(self.h1_val, self.h2_val, self.iter_val)
        
        self.canvas_size = int(self.h * 0.33)
        self.grid_cell = self.canvas_size // 28
        self.canvas_rect = pygame.Rect((self.w//2 - self.canvas_size//2), int(self.h * 0.65), self.canvas_size, self.canvas_size)
        
        self.bg_color, self.grid_color, self.accent_color = (4, 6, 8), (15, 20, 30), (0, 255, 170)
        self.dim_color, self.text_color, self.alert_color = (25, 35, 45), (140, 160, 180), (255, 40, 40)

        self.btn_opt = pygame.Rect(30, self.h - 70, 120, 45)
        self.btn_info = pygame.Rect(160, self.h - 70, 45, 45)
        self.panel_rect = pygame.Rect(30, self.h - 520, 360, 440)
        
        self.btn_h1_sub, self.btn_h1_add = pygame.Rect(220, self.panel_rect.y+60, 30, 30), pygame.Rect(310, self.panel_rect.y+60, 30, 30)
        self.btn_h2_sub, self.btn_h2_add = pygame.Rect(220, self.panel_rect.y+110, 30, 30), pygame.Rect(310, self.panel_rect.y+110, 30, 30)
        self.btn_it_sub, self.btn_it_add = pygame.Rect(220, self.panel_rect.y+160, 30, 30), pygame.Rect(310, self.panel_rect.y+160, 30, 30)
        self.btn_apply = pygame.Rect(50, self.panel_rect.y+210, 290, 45)
        self.btn_model_prev, self.btn_model_next = pygame.Rect(50, self.panel_rect.y+320, 35, 35), pygame.Rect(305, self.panel_rect.y+320, 35, 35)
        self.btn_load = pygame.Rect(50, self.panel_rect.y+370, 290, 45)

    def draw_ui(self):
        self.screen.fill(self.bg_color)
        for x in range(0, self.w, 60): pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, 60): pygame.draw.line(self.screen, self.grid_color, (0, y), (self.w, y), 1)
        self.screen.blit(self.font_md.render("[ESC] EXIT", True, self.alert_color), (self.w - 140, 20))

        pygame.draw.rect(self.screen, (2, 4, 6), self.canvas_rect)
        for y in range(28):
            for x in range(28):
                if self.grid[y, x] > 0:
                    c = int(self.grid[y, x] * 255)
                    pygame.draw.rect(self.screen, (0, c, int(c*0.6)), (self.canvas_rect.x + x*self.grid_cell, self.canvas_rect.y + y*self.grid_cell, self.grid_cell, self.grid_cell))
        pygame.draw.rect(self.screen, self.accent_color, self.canvas_rect, 1)

        pygame.draw.rect(self.screen, self.dim_color, self.btn_opt, 1)
        self.screen.blit(self.font_md.render("OPTIONS", True, self.accent_color if self.show_settings else self.text_color), (self.btn_opt.x+20, self.btn_opt.y+12))
        pygame.draw.rect(self.screen, self.dim_color, self.btn_info, 1)
        self.screen.blit(self.font_md.render("?", True, self.accent_color if self.show_info else self.text_color), (self.btn_info.x+15, self.btn_info.y+12))

        if self.show_info:
            p = pygame.Rect(160, self.h - 140, 420, 65)
            pygame.draw.rect(self.screen, (10, 15, 25), p); pygame.draw.rect(self.screen, self.accent_color, p, 1)
            self.screen.blit(self.font_sm.render("LMB: WRITE | RMB: PURGE | [C]: CLEAR SURFACE", True, self.text_color), (p.x+15, p.y+22))

        if self.show_settings:
            pygame.draw.rect(self.screen, (8, 12, 18), self.panel_rect); pygame.draw.rect(self.screen, self.accent_color, self.panel_rect, 1)
            self.screen.blit(self.font_md.render("NEURAL ARCHITECTURE", True, self.accent_color), (50, self.panel_rect.y+15))
            cfg = [("L2 WIDTH", self.h1_val, self.btn_h1_sub), ("L3 WIDTH", self.h2_val, self.btn_h2_sub), ("MAX ITER", self.iter_val, self.btn_it_sub)]
            for lbl, v, b in cfg:
                self.screen.blit(self.font_sm.render(lbl, True, self.text_color), (50, b.y+8))
                pygame.draw.rect(self.screen, self.dim_color, b, 1); pygame.draw.rect(self.screen, self.dim_color, (b.x+90, b.y, 30, 30), 1)
                self.screen.blit(self.font_md.render("-", True, self.accent_color), (b.x+10, b.y+5)); self.screen.blit(self.font_md.render("+", True, self.accent_color), (b.x+100, b.y+5))
                self.screen.blit(self.font_md.render(str(v), True, self.accent_color), (b.x+40, b.y+5))
            pygame.draw.rect(self.screen, self.dim_color, self.btn_apply, 1)
            self.screen.blit(self.font_md.render("RE-TRAIN ENGINE", True, self.accent_color), (self.btn_apply.x+60, self.btn_apply.y+12))
            self.screen.blit(self.font_md.render("LOCAL WEIGHTS", True, self.accent_color), (50, self.panel_rect.y+285))
            pygame.draw.rect(self.screen, self.dim_color, self.btn_model_prev, 1); pygame.draw.rect(self.screen, self.dim_color, self.btn_model_next, 1)
            self.screen.blit(self.font_md.render("<", True, self.accent_color), (self.btn_model_prev.x+12, self.btn_model_prev.y+6))
            self.screen.blit(self.font_md.render(">", True, self.accent_color), (self.btn_model_next.x+12, self.btn_model_next.y+6))
            if self.saved_models:
                m = self.saved_models[self.selected_model_idx]
                self.screen.blit(self.font_sm.render(f"H:{m[0]}-{m[1]} I:{m[2]}", True, self.accent_color), (self.btn_model_prev.x+55, self.btn_model_prev.y+10))
            pygame.draw.rect(self.screen, self.dim_color, self.btn_load, 1)
            self.screen.blit(self.font_md.render("DEPLOY SELECTED", True, self.accent_color), (self.btn_load.x+70, self.btn_load.y+12))

    def draw_network(self):
        if not self.nn.activations: return
        labels = ["L1: SENSOR", "L2: HIDDEN", "L3: HIDDEN", "L4: OUTPUT"]
        margin = int(self.w * 0.12)
        layer_x = [margin + i * ((self.w - 2*margin) // 3) for i in range(4)]
        n_h, y_o = int(self.h * 0.58), int(self.h * 0.06)
        
        for i in range(3):
            act_c, act_n = self.nn.activations[i], self.nn.activations[i+1]
            if i == 0: idx_c = np.linspace(0, len(act_c)-1, 32, dtype=int)
            else: idx_c = np.arange(len(act_c))
            
            idx_n = np.arange(len(act_n))
            sp_c, sp_n = n_h / max(1, len(idx_c)), n_h / max(1, len(idx_n))
            ys_c, ys_n = y_o + (n_h - len(idx_c)*sp_c)/2, y_o + (n_h - len(idx_n)*sp_n)/2
            
            line_limit = 5000 
            line_count = 0
            for n1, i1 in enumerate(idx_c):
                if act_c[i1] > 0.15:
                    y1 = ys_c + n1*sp_c
                    for n2, i2 in enumerate(idx_n):
                        if line_count < line_limit:
                            pygame.draw.line(self.screen, (25, 40, 50), (layer_x[i], y1), (layer_x[i+1], ys_n + n2*sp_n), 1)
                            line_count += 1

        for i, x in enumerate(layer_x):
            act = self.nn.activations[i]
            if i == 0: idx_list = np.linspace(0, len(act)-1, 32, dtype=int)
            else: idx_list = np.arange(len(act))
            
            sp = n_h / max(1, len(idx_list))
            ys = y_o + (n_h - len(idx_list)*sp)/2
            r = max(1, min(6, int(sp // 2) - 1))
            self.screen.blit(self.font_sm.render(labels[i], True, self.accent_color), (x - 40, y_o - 25))
            
            for j, idx in enumerate(idx_list):
                v = float(act[idx])
                color = (int(v*self.accent_color[0]+(1-v)*self.dim_color[0]), int(v*self.accent_color[1]+(1-v)*self.dim_color[1]), int(v*self.accent_color[2]+(1-v)*self.dim_color[2]))
                pygame.draw.circle(self.screen, color, (x, int(ys + j*sp)), r)
                if v > 0.6: pygame.draw.circle(self.screen, self.accent_color, (x, int(ys + j*sp)), r+2, 1)
                if i == 3:
                    self.screen.blit(self.font_md.render(f"[{idx}]", True, self.accent_color if v > 0.5 else self.dim_color), (x+15, int(ys+j*sp-10)))
                    if v > 0.1: self.screen.blit(self.font_xs.render(f"{v*100:04.1f}%", True, self.text_color), (x+55, int(ys+j*sp-5)))

    def handle_input(self):
        p, b = pygame.mouse.get_pos(), pygame.mouse.get_pressed()
        if self.canvas_rect.collidepoint(p) and not self.show_settings:
            x, y = (p[0]-self.canvas_rect.x)//self.grid_cell, (p[1]-self.canvas_rect.y)//self.grid_cell
            if b[0]:
                self.grid[y, x] = 1.0
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    if 0<=y+dy<28 and 0<=x+dx<28: self.grid[y+dy, x+dx] = min(1.0, self.grid[y+dy, x+dx]+0.5)
            elif b[2]: self.grid[y, x] = 0.0

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_c: self.grid = np.zeros((28, 28))
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    m = pygame.mouse.get_pos()
                    if self.btn_opt.collidepoint(m): self.show_settings = not self.show_settings
                    elif self.btn_info.collidepoint(m): self.show_info = not self.show_info
                    elif self.show_settings:
                        if self.btn_h1_sub.collidepoint(m): self.h1_val = max(8, self.h1_val-8)
                        elif self.btn_h1_add.collidepoint(m): self.h1_val = min(512, self.h1_val+8)
                        elif self.btn_h2_sub.collidepoint(m): self.h2_val = max(8, self.h2_val-8)
                        elif self.btn_h2_add.collidepoint(m): self.h2_val = min(512, self.h2_val+8)
                        elif self.btn_it_sub.collidepoint(m): self.iter_val = max(50, self.iter_val-50)
                        elif self.btn_it_add.collidepoint(m): self.iter_val = min(2000, self.iter_val+50)
                        elif self.btn_apply.collidepoint(m): self.is_training = True
                        elif self.saved_models:
                            if self.btn_model_prev.collidepoint(m): self.selected_model_idx = (self.selected_model_idx-1)%len(self.saved_models)
                            elif self.btn_model_next.collidepoint(m): self.selected_model_idx = (self.selected_model_idx+1)%len(self.saved_models)
                            elif self.btn_load.collidepoint(m): self.is_loading = True
            if self.is_training or self.is_loading:
                self.screen.fill(self.bg_color)
                self.screen.blit(self.font_lg.render("NEURAL CORE SYNCING...", True, self.accent_color), (self.w//2-250, self.h//2))
                pygame.display.flip()
                if self.is_training: self.nn = NeuralNetwork(self.h1_val, self.h2_val, self.iter_val, True)
                else:
                    c = self.saved_models[self.selected_model_idx]
                    self.h1_val, self.h2_val, self.iter_val = c[0], c[1], c[2]
                    self.nn = NeuralNetwork(self.h1_val, self.h2_val, self.iter_val)
                self.saved_models, self.grid, self.is_training, self.is_loading, self.show_settings = get_saved_models(), np.zeros((28, 28)), False, False, False
            else:
                self.handle_input()
                self.nn.forward(center_image(self.grid).flatten())
                self.draw_ui(); self.draw_network()
                pygame.display.flip(); self.clock.tick(60)

if __name__ == "__main__": App().run()

import random
import joblib
import numpy as np
import statistics
import scipy
from scipy import signal
import scipy.integrate as integrate
from scipy.signal import find_peaks
from pywt import wavedec
import sys
from .world import World
from .settings import TILE_SIZE
from .utils import *

class Game:
    def __init__(self,screen,clock):
        self.screen = screen
        self.clock = clock
        self.width, self.height = self.screen.get_size()
        self.world = World(11,11,self.width, self.height)
        self.reserved_locations = [(5,5),(0,5),(10,5),(5,0),(5,10)]
        self.obstacles = []
        self.buildings_locations = {
            "N": (0,5),
            "S":(10,5),
            "E":(5,0),
            "W":(5,10)
        }
        self.player_pos = self.reserved_locations[0]
        self.deliveries = []
        for i in range(3):
            idx = random.randint(0,3)
            self.deliveries.append(list(self.buildings_locations.keys())[idx])
        # print(self.deliveries)
        self.cash = 0
        self.deliveries_made = 0
        self.RFC = joblib.load("../RandomForestClassifier.sav")
        self.ETC = joblib.load("../ExtraTreesClassifier.sav")
        self.VC = joblib.load("../VotingClassifier.sav")
        self.class_map = {
            "u":"yukari",
            "d":"asagi",
            "r":"sag",
            "l":"sol",
            "b":"kirp"
        }
        self.path = r'../3-class-clean'
        b, a = signal.butter(2, [1, 25], btype='band', analog=False, output='ba', fs=176)
        self.b = b
        self.a = a

    def run(self):
        self.playing = True
        while self.playing:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    dir = None
                    if event.key == pg.K_ESCAPE:
                        pg.quit()
                        sys.exit()
                    if event.key == pg.K_UP:
                        dir = self.movedir("u")
                    if event.key == pg.K_DOWN:
                        dir = self.movedir("d")
                    if event.key == pg.K_RIGHT:
                        dir = self.movedir("r")
                    if event.key == pg.K_LEFT:
                        dir = self.movedir("l")
                    if event.key == pg.K_RETURN:
                        target = self.check_relevancy()
                        if target:
                            dir = self.movedir("b")
                    if dir == "u":
                        self.move("N")
                    elif dir == "d":
                        self.move("S")
                    elif dir == "r":
                        self.move("E")
                    elif dir == "l":
                        self.move("W")
                    elif dir == "b":
                        self.make_delivery(target)
                    else:
                        continue

            self.events()
            self.draw()
    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()


            pg.display.update()
    def draw(self):
        self.screen.fill((0,0,0))

        self.screen.blit(self.world.grass_tiles, (0,0))

        for x in range(self.world.grid_length_x):
            for y in range(self.world.grid_length_y):
                p = self.world.world[x][y]["iso_poly"]
                p = [(x + self.width / 2, y + self.height / 16) for x, y in p]
                pg.draw.polygon(self.screen, (20, 150, 10), p, 1)
                render_pos = self.world.world[x][y]["render_pos"]
                if (x,y) == self.player_pos:
                    tile = "man"
                    self.screen.blit(self.world.tiles[tile], (render_pos[0] + self.width / 2 + 10,render_pos[1] - 15 + self.height / 16 - (self.world.tiles[tile].get_height() - TILE_SIZE)))
                    offset = 120
                    for delivery in self.deliveries:
                        tile = delivery
                        self.screen.blit(self.world.tiles[tile], (render_pos[0] + self.width / 2 - 20,render_pos[1] - offset + self.height / 16 - (self.world.tiles[tile].get_height() - TILE_SIZE)))
                        offset -= 40

                if (x,y) in self.reserved_locations[1:3]:
                    self.obstacles.append((x, y))
                    tile = "building1"
                    self.screen.blit(self.world.tiles[tile], (render_pos[0] + self.width / 2,render_pos[1] + self.height / 16 - (self.world.tiles[tile].get_height() - TILE_SIZE)))
                if (x,y) in self.reserved_locations[3:5]:
                    self.obstacles.append((x, y))
                    tile = "building2"
                    self.screen.blit(self.world.tiles[tile], (render_pos[0] + self.width / 2,render_pos[1] + self.height / 16 - (self.world.tiles[tile].get_height() - TILE_SIZE)))

                tile = self.world.world[x][y]["tile"]
                if tile != "":
                    if (x,y) in self.reserved_locations:
                        continue
                    else:
                        self.obstacles.append((x, y))
                        if tile == "tree":
                            self.screen.blit(self.world.tiles[tile], (render_pos[0] + self.width / 2 ,render_pos[1] -20 + self.height / 16 - (self.world.tiles[tile].get_height() - TILE_SIZE)))
                        else:
                            self.screen.blit(self.world.tiles[tile], (render_pos[0] + self.width / 2,render_pos[1] + self.height / 16 - (self.world.tiles[tile].get_height() - TILE_SIZE)))

        draw_text(
            self.screen,
            'current deliveries = [{},{},{}]'.format(self.deliveries[0],self.deliveries[1],self.deliveries[2]),
            25,
            (255, 255, 255),
            (10, 50)
        )
        draw_text(
            self.screen,
            'correct deliveries made = {}'.format(self.deliveries_made),
            25,
            (255, 255, 255),
            (10, 100)
        )
        draw_text(
            self.screen,
            'cash = {}'.format(self.cash),
            25,
            (255, 255, 255),
            (10, 150)
        )

        pg.display.flip()
    def preprocess_signal(self,np_signal):
        filtered = signal.filtfilt(self.b, self.a, np_signal)
        resampled = signal.resample(filtered, 60)
        mean = statistics.mean(resampled)
        dc_removed = np.array([(sample - mean) for sample in resampled])
        norm = np.linalg.norm(dc_removed)
        normalized = dc_removed / norm
        return normalized
    def pick_random_file(self,direction):
        idx = random.randint(1, 20)
        temp = self.class_map[direction]
        file1 = self.path + r"/" + temp + str(idx) + 'h.txt'
        file2 = self.path + r"/" + temp + str(idx) + 'v.txt'
        h = np.loadtxt(file1, dtype=int)
        v = np.loadtxt(file2, dtype=int)
        return (self.preprocess_signal(h), self.preprocess_signal(v))
    def analyze_w_wavelet(self, np_signal):
        coefs = wavedec(np_signal, 'db4', level=3)
        features = [coefs[0], coefs[1], coefs[2]]
        return features
    def peak_features(self, signal):
        peaks = find_peaks(signal)
        Y = []
        for i in range(len(peaks) - 1):
            Y.append(signal[i])
        return max(Y)
    def area_features(self,signal):
        feature = integrate.simps(signal)
        return feature
    def psd_features(self,signal, fs):
        (f, S) = scipy.signal.periodogram(signal, fs, scaling='density')
        return S
    def extract_features(self,h, v):
        features = []
        features.extend([item for sublist in self.analyze_w_wavelet(h) for item in sublist])
        features.extend(self.psd_features(h, 50))
        features.extend([item for sublist in self.analyze_w_wavelet(v) for item in sublist])
        features.extend(self.psd_features(v, 50))
        features.append(self.area_features(h))
        features.append(self.peak_features(h))
        features.append(self.area_features(v))
        features.append(self.peak_features(v))
        return features
    def classify(self,x):
        pred1 = self.RFC.predict(np.array(x).reshape(1, -1))
        pred2 = self.ETC.predict(np.array(x).reshape(1, -1))
        if pred1 == pred2:
            print("RFC & ETC result: " + pred1)
            return pred1
        else:
            pred3 = self.VC.predict(np.array(x).reshape(1, -1))
            print("VC result: " + pred3)
            return pred3
    def movedir(self,key):
        h, v = self.pick_random_file(key)
        direction = self.classify(self.extract_features(h, v))
        return direction
    def check_relevancy(self):
        locs = list(self.buildings_locations.keys())
        for i in range(len(locs)):
            pos = self.buildings_locations[locs[i]]
            if self.player_pos[0] in range(pos[0] - 1, pos[0] + 2) and self.player_pos[1] in range(pos[1] - 1, pos[1] + 2):
                return locs[i]
        return None
    def make_delivery(self, location):
        if location == self.deliveries[0]:
            self.cash += 100
            self.deliveries_made += 1
        else:
            self.cash -= 100
        self.deliveries.pop(0)
        self.deliveries.append(self.deliveries.append(list(self.buildings_locations.keys())[random.randint(0,3)]))
        self.deliveries.pop(-1)
        print("cash:{}".format(self.cash))
        print("deliveries:{}".format(self.deliveries_made))
        print(self.deliveries)
    def move(self, direction):
        next_loc = None

        if direction == "N":
            next_loc = (self.player_pos[0] - 1, self.player_pos[1])
        elif direction == "S":
            next_loc = (self.player_pos[0] + 1, self.player_pos[1])
        if direction == "E":
            next_loc = (self.player_pos[0], self.player_pos[1] - 1)
        elif direction == "W":
            next_loc = (self.player_pos[0], self.player_pos[1] + 1)
        if next_loc[0] < 0 or next_loc[0] > 10 or next_loc[1] < 0 or next_loc[1] > 10:
            return
        for loc in self.obstacles:
            if next_loc == loc:
                return
        self.player_pos = next_loc
import pygame as pg
import random
from .settings import TILE_SIZE

class World:
    def __init__(self,grid_length_x, grid_length_y, width, height):
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.width = width
        self.height = height
        self.tiles = self.load_images()
        self.grass_tiles = pg.Surface((width, height))
        self.world = self.create_world()

    def create_world(self):
        world = []

        for grid_x in range(self.grid_length_x):
            world.append([])
            for grid_y in range(self.grid_length_y):
                world_tile = self.grid_to_world(grid_x,grid_y)
                world[grid_x].append(world_tile)

                render_pos = world_tile["render_pos"]
                self.grass_tiles.blit(self.tiles["block"], (render_pos[0] + self.width / 2, render_pos[1] + self.height/16))

        return world
    def grid_to_world(self, grid_x, grid_y):
        rect = [
            (grid_x * TILE_SIZE, grid_y * TILE_SIZE),
            (grid_x * TILE_SIZE + TILE_SIZE, grid_y * TILE_SIZE),
            (grid_x * TILE_SIZE + TILE_SIZE, grid_y * TILE_SIZE + TILE_SIZE),
            (grid_x * TILE_SIZE, grid_y * TILE_SIZE + TILE_SIZE)
        ]

        iso_poly = [self.cart_to_iso(x,y) for x,y in rect]

        minx = min([x for x,y in iso_poly])
        miny = min([y for x,y in iso_poly])

        r = random.randint(1,100)
        if r <= 5:
            tile = "tree"
        elif r <= 10:
            tile = "rock"
        else:
            tile = ""

        out = {
            "grid" : [grid_x, grid_y],
            "cart_rect" : rect,
            "iso_poly": iso_poly,
            "render_pos": [minx,miny],
            "tile": tile
        }

        return out
    def cart_to_iso(self,x,y):
        iso_x = x - y
        iso_y = (x + y) / 2
        return iso_x,iso_y
    def load_images(self):
        man = pg.image.load("assets/graphics/man.png")
        block = pg.image.load("assets/graphics/block.png")
        rock = pg.image.load("assets/graphics/rock.png")
        tree = pg.image.load("assets/graphics/tree.png")
        building1 = pg.image.load("assets/graphics/building01.png")
        building2 = pg.image.load("assets/graphics/building02.png")
        N = pg.image.load("assets/graphics/Nboxs.png")
        S = pg.image.load("assets/graphics/Sboxs.png")
        E = pg.image.load("assets/graphics/Eboxs.png")
        W = pg.image.load("assets/graphics/Wboxs.png")

        return {
            "man": man,
            "block": block,
            "rock": rock,
            "tree": tree,
            "building1": building1,
            "building2": building2,
            "N":N,
            "S":S,
            "E":E,
            "W":W
        }
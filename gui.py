from tkinter import *
import math
import winsound



POLYGON_SIZE = 30
WIDTH =1000
HEIGH = 900


root = Tk()
root.title("Hex")
canvas = Canvas(root, width=1000, height=900)
canvas.pack()

#canvas.create_polygon(100,150, 150,200, 200,200, 200,150, fill = "black")

def get_hex_corner_coord(center, i):
    angle_deg = 60*i + 30
    angle_rad = math.pi*angle_deg/180
    x = center[0] + POLYGON_SIZE*math.cos(angle_rad)
    y = center[1] + POLYGON_SIZE*math.sin(angle_rad)
    return x, y

def draw_polygon(center):
    coords = [get_hex_corner_coord(center, i) for i in range(6)]
    canvas.create_polygon(coords, fill = "grey", outline = "black")
   

def board_index_to_center(index):
    """
    finds coordinates of the polygon center based on board index
    """
    i = index[0]
    j = index[1]
    x = WIDTH/2 + j*POLYGON_SIZE*math.sqrt(3)/2 - i*POLYGON_SIZE*math.sqrt(3)/2 
    y = HEIGH/20 + j*POLYGON_SIZE*3/2+i*POLYGON_SIZE*3/2
    return (x,y)


def draw_grid(size):
    for i in range(size):
         for j in range(size):
            draw_polygon(board_index_to_center((i,j)))

draw_grid(10)


def on_canvas_click(event):
    item = canvas.find_closest(event.x, event.y)
    print(item[0])
    canvas.itemconfig(item, fill='red')
    winsound.Beep(200, 20)

canvas.bind('<Button-1>', on_canvas_click)

root.mainloop()

class Hex:
    """
    Main class for the Hex Game interface
    """

    def __init__(self, root = Tk(), size=3) -> None:
        self.root = root


    def game_loop(self):
        self.root.mainloop()

from tkinter import *
import math
import winsound
   

class Hex:
    """
    Main class for the Hex Game interface
    """
    POLYGON_SIZE = 30
    WIDTH =1000
    HEIGH = 900

    def __init__(self, root = Tk(), size=3, board_view = None) -> None:
        self.root = root
        self.size=size
        self.board_view = board_view # will be fixed by the Controller
        self.listener = None
       
        

    def game_loop(self):
        self.canvas = Canvas(self.root, width=Hex.WIDTH, heigh = Hex.HEIGH)
        self.canvas.pack()
        self.draw_grid()
        #self.canvas.bind('<Button-1>', self.listener.on_canvas_click)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        self.root.title("Hex")
        self.root.mainloop()

    def get_hex_corner_coord(self, center, i):
        angle_deg = 60*i + 30
        angle_rad = math.pi*angle_deg/180
        x = center[0] + Hex.POLYGON_SIZE*math.cos(angle_rad)
        y = center[1] + Hex.POLYGON_SIZE*math.sin(angle_rad)
        return x, y
        
    def draw_polygon(self, center):
        coords = [self.get_hex_corner_coord(center, i) for i in range(6)]
        self.canvas.create_polygon(coords, fill = "grey", outline = "black")

    def board_index_to_center(self, index):
        """
        finds coordinates of the polygon center based on board index
        """
        i = index[0]
        j = index[1]
        x = Hex.WIDTH/2 + j*Hex.POLYGON_SIZE*math.sqrt(3)/2 - i*Hex.POLYGON_SIZE*math.sqrt(3)/2 
        y = Hex.HEIGH/20 + j*Hex.POLYGON_SIZE*3/2+i*Hex.POLYGON_SIZE*3/2
        return (x,y)


    def draw_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                self.draw_polygon(self.board_index_to_center((i,j)))

    def on_canvas_click(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        #print(item[0])
        self.canvas.itemconfig(item, fill='red')
        winsound.Beep(200, 20)
        self.listener.on_canvas_click(event, item)
    
    def add_listener(self, listener):
        self.listener = listener


#hex = Hex(size=10)
#hex.game_loop()
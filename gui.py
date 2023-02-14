from tkinter import *
import math
import winsound
   

class Hex:
    """
    Main class for the Hex Game interface
    """
    
    WIDTH =1000
    HEIGH = 900
    POLYGON_SIZE = 30

    def __init__(self, root = Tk(), size=3, board_view = None) -> None:
        self.root = root
        self.size=size
        self.board_view = board_view # will be fixed by the Controller
        self.listener = None
       
        

    def game_loop(self):
        self.canvas = Canvas(self.root, width=Hex.WIDTH, heigh = Hex.HEIGH)
        self.canvas.pack()
        self.draw_grid()
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.root.title("Hex")
        self.player_label = Label(text = "BLACK's turn", font=36)
        self.player_label.place(x=20, y=40)
        self.won_label = Label(text = "", font=72)
        self.won_label.place(x=20, y=80)
        self.root.mainloop()

    def get_hex_corner_coord(self, center, i):
        angle_deg = 60*i + 30
        angle_rad = math.pi*angle_deg/180
        x = center[0] + Hex.POLYGON_SIZE*math.cos(angle_rad)
        y = center[1] + Hex.POLYGON_SIZE*math.sin(angle_rad)
        return x, y
        
    def draw_polygon(self, center, fill):
        coords = [self.get_hex_corner_coord(center, i) for i in range(6)]
        self.canvas.create_polygon(coords, fill = fill, outline = "#777777")

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
                if i == 0 and j == self.size-1 or i == 0 and j == 0 or i == self.size-1 and j == self.size-1 or i == self.size-1 and j == 0:
                    fill = "#998888"
                elif j == 0 or j == self.size-1:
                    fill = "#bbaaaa" #red ish
                elif i == 0 or i == self.size-1:
                    fill = "#999999" # black ish
                else:
                    fill = "#aaaaaa"
                self.draw_polygon(self.board_index_to_center((i,j)), fill = fill)
    

    def on_canvas_click(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        self.listener.on_canvas_click(event, item)
    
    def add_listener(self, listener):
        self.listener = listener

    def end_game(self):
        self.canvas.unbind('<Button-1>')

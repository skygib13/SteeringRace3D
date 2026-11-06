
from ursina import *

app = Ursina()

window.title = "3D TEST"
window.color = color.rgb(70, 140, 200)

ground = Entity(
    model="cube",
    color=color.green,
    scale=(20, 0.2, 20),
    position=(0, -2, 0)
)

car = Entity(
    model="cube",
    color=color.red,
    scale=(2, 1, 4),
    position=(0, 0, 0)
)

camera.position = (0, 5, -15)
camera.look_at(car)

Sky()

app.run()

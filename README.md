There are two python codes in here. They are:

* cubic_realroots.py: this code just finds the real roots of a cubic polynomial, and has nothing to do with physics or black holes. However, as it turns out, determining the real roots a cubic equation is a first step towards finding geodesics in Schwarzschild spacetime! This code mainly follows the prescriptions detailed in the wikipedia page for cubic equation.

* sch_null.py: this code calculates, and then optional plots the geodesics. I have tried to add some documentation at the beginning of the code, describing how to use it. The code uses prescriptions described in Chandrasekhar's book "The Mathematical Theory of Black Holes", and the book "Gravitation" by Misner, Thorne, and Wheeler. I have added a few example cases at the bottom of the code. The cubic_realroots.py need to be in the same directory as this code, for this code to run properly.

Other than basic python, these codes need numpy and matplotlib modules. 

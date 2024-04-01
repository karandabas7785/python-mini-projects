"""
Co-ordinates Module -

###################################################################################################################



"""

import matplotlib.pyplot as plt
import math
import sympy

import numpy as np

sympy.init_printing()

class Point:

    """
    Point creates the x,y co-ordinate objects on the xy plane
    returns (x,y) co-ordinates
    """
    def __init__(self,x,y):
        self.__x=x
        self.__y=y
    
    def __str__(self):
        return f"({self.__x},{self.__y})"
    
    def getXYloc(self):
        """
        returns the x,y location of the point
        """
        return self.__x,self.__y

    def setXYloc(self,x,y):
        """
        sets the x,y location of the point
        """
        self.__x=x
        self.__y=y
    
    def findMidpoint(self,other):
        """
        returns the point of midpoint of 2 given points
        """
        x1,y1=self.getXYloc()
        x2,y2=other.getXYloc()
        x=(x1+x2)/2
        y=(y1+y2)/2
        return Point(x,y)
    
    def findSectionPoint(self,other,m,n):
        """
        returns the point of SectionPoint of 2 given points
        m,n is the ratio as m:n from self to other
        """
        x1,y1=self.getXYloc()
        x2,y2=other.getXYloc()
        x=(n*x1+m*x2)/(m+n)
        y=(n*y1+m*y2)/(m+n)
        return Point(x,y)

    def getIntegralPoints(self,other):
        #https://www.geeksforgeeks.org/number-integral-points-two-points/?ref=lbp
        pass
    
    def getLine2PointForm(self,other):
        x1,y1=self.getXYloc()
        x2,y2= other.getXYloc()
        m= (y2-y1)/(x2-x1)
        x,y=sympy.symbols('x y')
        e= sympy.Eq((y-y1)-m*x-m*x1,0)
        return e
    
    def getLinePointSlopeform(self,m):
        x1,y1=self.getXYloc()
        x,y=sympy.symbols('x y')
        e= sympy.Eq((y-y1)-m*x-m*x1,0)
        return e
    

class distance:
    """
    def distance(obj1:Point,obj2:Point):

    Returns the distance between two Point objects
    """
    def __init__(self,obj1:Point,obj2:Point):
        """
        Takes 2 objects of Point class defining 2 co-ordinates
        """
        self.__x1,self.__y1=obj1.getXYloc()
        self.__x2,self.__y2=obj2.getXYloc()

    def __str__(self):
        result=self.__calculate_distance()
        #print(type(result))
        return str(result)
    
    def getDistance(self):
        """
        Returns the distance between two Point object
        """
        return self.__calculate_distance

    def setDistance(self,obj1:Point,obj2:Point):
        """
        def setDistance(self,obj1,obj2)

        To set the Point objects in distance 
        """
        self.__x1,self.__y1=obj1.getXYloc()
        self.__x2,self.__y2=obj2.getXYloc()

    
    def __calculate_distance(self):
        """
        Calculates the distance between 2 points
        """
        x=math.pow(self.__x2- self.__x1,2)
        y=math.pow(self.__y2- self.__y1,2)
        return math.sqrt(x+y)

# obj1=Point(4,5)
# obj2=Point(3,5)
# print (distance(obj1,obj2))

class Shape:
    """
    defines the shape
    """
    def __init__(self,shape):
        self.__shape=shape
    
    def __str__(self):
        return self.__shape
    
    def getShape(self):
        """
        Returns the shape of the object
        """
        return self.__shape
    def setShape(self,shape):
        """
        Sets the shape of the object
        """
        self.__shape=shape
    
    def area(self):
        """
        Calculates the area
        """
        raise NotImplementedError("area() is not yet implemented")
    
    def draw(self):
        """
        Draw the shape
        """
        raise NotImplementedError("draw() is not yet implemented")

    def resize(self,factor):
        """
        Resize the shape
        """
        raise NotImplementedError("resize() is not yet implemented")

    def getEquation(self):
        """
        Returns the equation of the shape
        """
        raise NotImplementedError("getEquation() is not yet implemented")


class CircleRadiusForm(Shape):

    """
    Contains all the functions of Circle
    """
    def __init__(self,r,origin:Point):
        """
        accepts r and origin- a Point Object
        """
        Shape.__init__(self,"Circle")
        self.__shape=self.getShape()
        self.__x,self.__y= origin.getXYloc()
        self.__radius = r
    
    def __str__(self):
        return str(self.getEquation())

    def area(self):
        return math.pi* (self.__radius**2)

    def getCirle(self):
        return self.__radius,Point(self.__x,self.__y)

    def setCircle(self,radius,origin:Point):
        Shape.__init__(self,"Circle")
        self.__shape=self.getShape()
        self.__x,self.__y= origin.getXYloc()
        self.__radius = radius



    def draw(self):
        #x2 + y2 + 2gx +2fy +c=0
        angle = np.linspace( 0 , 2 * np.pi , 300) 
 
        radius = self.__radius
        
        x = self.__x+(radius * np.cos( angle ))
        y = self.__y+(radius * np.sin( angle )) 
        #print(x,y)
        axes=plt.gca()
        axes.set_aspect( 1 ,adjustable='datalim') 
        
        axes.plot( x, y ) 
         

        #print(plt)
        plt.title( 'Circle' ) 
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        
        plt.show() 
    
    def resize(self,factor):
        """
        Resize function resizes the graph. 
        Factor is 1 for no change
        Factor is greater than 1 for making it large
        Factor is smaller than 1 for making it small
        """
        self.__radius*= factor
    
    def getEquation(self):
        """
        Returns the equation of the Circle
        """
        #x2 + y2 + 2gx +2fy +c=0
        a=1
        b=1
        r=self.__radius
        g=self.__x
        f=self.__y
        c= (g**2 + f**2 - r**2)
        sympy.init_printing()
        y, x = sympy.symbols(('y', 'x'))
        #ax^2 + by^2 + 2gx + 2fy +c=0
        
        
        return sympy.Eq(x**2 + y**2 + 2*g*x + 2*f*y +c,0)


class TwoDegreeEquation(Shape):
    def __init__(self,*args):
        """
        Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0

        Takes only 6 args - in order as a,h,b,g,f,c
        """
        #Shape.__init__(self,"Circle")
        if len(args)==6:
            self.a=args[0]
            self.h=args[1]
            self.b=args[2]
            self.g=args[3]
            self.f=args[4]
            self.c=args[5]
            self.arg=args
            Shape.__init__(self,self.checkShape())
        else:
            raise TypeError("Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0 \n\n \
            TwoDegreeEquation takes only 6 args - in order as a,h,b,g,f,c")

    def setValue(self, *args):
        self.__init__(*args)
        
    def __str__(self):
        return str(self.getEquation())

    def getEquation(self):
        """
        Returns the mathematical equation of any 2 degree equation
        """
        y, x = sympy.symbols(('y', 'x'))
        #ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0
        sympy.init_printing()

        return sympy.Eq((self.a*(x**2)) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x)+(2*self.f*y)+ self.c,0)

    def checkShape(self):
        """
        Checks the shape of 
        """
        #abc + 2fgh – af2 – bg2 – ch2 
        k= (self.a*self.b*self.c) + (2*self.f*self.g*self.h) - (self.a*(self.f**2))-(self.b*(self.g**2))-(self.c*(self.h**2) )
        #print (k)
        if round(k)==0:
            return "PairOfStraightLine"
        elif self.a==self.b and self.h==0 and self.g**2 + self.f**2 - self.c>0:
            return "Circle"
        elif self.a==self.b and self.h==0 and self.g**2 + self.f**2 - self.c==0:
            return "Point"
        elif self.h**2 == self.a*self.b:
            return "Parabola"
        elif self.h**2 < self.a*self.b:
            return "Ellipse"
        elif self.h**2 > self.a*self.b:
            return "Hyperbola"
        else:
            return
    
    def draw(self):
        x = np.linspace(-1000, 1000, 300)
        y = np.linspace(-1000, 1000, 300)
        X, Y = np.meshgrid(x, y)
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c

        fig,ax = plt.subplots()
        ax.contour(X, Y, F, levels=[0]) # take level set corresponding to 0
        plt.show()
    
    def isPointOnCurve(self,X,Y):
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F==0 else False

    def isPointInside(self,X,Y):
        raise NotImplementedError(f"isPointInside() is not yet implemented to {self.checkShape()}")

    def isPointOutside(self,X,Y):
        raise NotImplementedError(f"isPointInside() is not yet implemented to {self.checkShape()}")
        

class PairOfStraightLine(TwoDegreeEquation):
    def __init__(self, *args):
        """
        Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0

        Takes only 6 args - in order as a,h,b,g,f,c
        """
        if len(args)==6:
            TwoDegreeEquation.__init__(self,*args)
            if self.getShape()!="PairOfStraightLine":
                raise ValueError("Not a Pair of Straight Line")
        else:
            raise TypeError("Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0 \n\n \
            TwoDegreeEquation takes only 6 args - in order as a,h,b,g,f,c")

    def setValue(self, *args):
        self.__init__(*args) 
    
    def __str__(self):
        return str(self.getEquation())

    def getEquation(self):
        """
        Returns the mathematical equation of any 2 degree equation
        """
        y, x = sympy.symbols(('y', 'x'))
        #ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0
        sympy.init_printing()
        return sympy.Eq((self.a*(x**2)) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x)+(2*self.f*y)+ self.c,0)

    def draw(self):
        x = np.linspace(-1000, 1000, 300)
        y = np.linspace(-1000, 1000, 300)
        X, Y = np.meshgrid(x, y)
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c

        fig,ax = plt.subplots()
        ax.contour(X, Y, F, levels=[0]) # take level set corresponding to 0
        plt.show()
    

    def getLines(self):
        a1,a2=self.getSlopes()
        m,n=sympy.symbols('m n')
        e1=sympy.Eq(-(a1*m+a2*n),2*self.g)
        e2=sympy.Eq((m+n),2*self.f)
        sol= sympy.solve((e1,e2),(m,n))
        c1=sympy.N(sol[m]);c2=sympy.N(sol[n])
        print(c1*c2==self.c)
        x,y=sympy.symbols('x y')
        l1= sympy.Eq(y-a1*x+c1,0)
        l2= sympy.Eq(y-a2*x+c2,0)
        return l1,l2


    def getSlopes(self):
        m1=sympy.symbols('m1')
        expression= sympy.Eq(((self.b**2)*(m1**2))+ (2*self.h*self.b*m1)+self.a,0)
        a=sympy.solve(expression,m1)
        
        m1,m2=a
        m3=self.a/(self.b*sympy.N(m1))
        print(m3)
        
        return sympy.N(m1),sympy.N(m2)
        
        

    def getAngle(self):

        if self.a+self.b==0:
            theta=90
        else:
            theta =math.degrees(math.tanh(2*math.sqrt(self.h**2 - self.a*self.b)/(self.a+self.b)))
        return theta

    def getPointOfIntersection(self):

        x,y=sympy.symbols('x y')
        expression= self.a*(x**2) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x)+(2*self.f*y)+ self.c  
        l1=sympy.Eq(sympy.diff(expression, y),0) #Family of line
        l2=sympy.Eq(sympy.diff(expression, x),0)
        sol=sympy.solve((l1,l2),(x,y))
        return Point(sol[x],sol[y])

    def getFamilyOfLines(self):
        a=self.getPointOfIntersection()
        x1,y1=a.getXYloc()
        m1,m2= self.getSlopes()
        x,y,m=sympy.symbols('x y m')
        l= sympy.Eq(y-m*x+m1*x1-y1,0) #family of lines 
        #y-y1=m(x-x1)
        #change the value of m and family of lines will appear
        return l

    def getAngleBisectors(self):
        m1,m2=self.getSlopes()
        a=self.getPointOfIntersection()
        x1,y1=a.getXYloc()
        x,y,m=sympy.symbols('x y m')
        e1= ((m2-m)/1+m2*m)
        e2=((m1-m)/1+m1*m)
        exp=sympy.Eq(e1,e2)
        sol1= sympy.solve(exp,m)
        #print(sol1[0])
        sol1=sol1[0]
        sol2=-1.0/sol1
        #print(sol2)
        l1=sympy.Eq(y-sol1*x+sol1*x1-y1,0)
        l2=sympy.Eq(y-sol2*x+sol2*x1-y1,0)
        return l1,l2
    

class StraightLine(Shape):
    def __init__(self,a,b,c):
        """
        Enter Line in format as ax+by+c=0
        """
        self.a=a
        self.b=b
        self.c=c
        try:
            self.slope=-a/b
        except ZeroDivisionError:
            self.slope=math.pow(10,10)
            #print(self.slope)
            #print(1/float("inf"))
        Shape.__init__(self,"Stright Line")

    def __str__(self):
        return str(self.getEquation())
    
    def getEquation(self):
        x,y=sympy.symbols('x y')
        return sympy.Eq(self.a*x+self.b*y+self.c,0)
    
    def getSlope(self):
        return self.slope
    
    def draw(self):
        x = np.linspace(-1000, 1000, 300)
        y = np.linspace(-1000, 1000, 300)
        X, Y = np.meshgrid(x, y)
        F = self.a*X+self.b*Y+self.c

        fig,ax = plt.subplots()
        ax.contour(X, Y, F, levels=[0]) # take level set corresponding to 0
        plt.show()

    def getAngle(self,line2):
        m1=line2.getSlope()
        m2=self.getSlope()
        if m1*m2!=1:
            O= abs((m2-m1)/(1+m1*m2))
            theta= math.degrees(math.tanh(O))
        else:
            theta=90
        return theta

    def isPointOnLine(self,X,Y):
        F = self.a*X+ self.b*Y+ self.c
        return True if F==0 else False

    def DistanceOfPointFromLine(self,point):
        x,y=point.getXYloc()
        a,b,c=self.a,self.b,self.c
        d= abs(a*x+b*y+c)/(math.sqrt(a**2+b**2))
        return d
    
    def FootOfPointOnLine(self,point):
        x,y=point.getXYloc()
        a,b,c=self.a,self.b,self.c
        d= -1*(abs(a*x+b*y+c)/(a**2+b**2))
        x1=x+a*d
        y1=y+b*d
        return Point(x1,y1)

    def ImageOfPointOfLine(self,point):
        x,y=point.getXYloc()
        a,b,c=self.a,self.b,self.c
        d= -2*(abs(a*x+b*y+c)/(a**2+b**2))
        x1=x+a*d
        y1=y+b*d
        return Point(x1,y1)

    def getPointOfIntersection(self,line):
        x,y=sympy.symbols('x y')
        l1= self.a*x+self.b*y+self.c
        l2= line.a*x+line.b*y+line.c
        sol=sympy.solve((l1,l2),(x,y))
        return Point(sol[x],sol[y])

    def getParallelLine(self,point):
        X,Y=point.getXYloc()
        slope=self.getSlope()
        x,y=sympy.symbols('x y')
        f=(y-Y)-(slope*(x-X))
        eq=sympy.Eq(f,0)
        return eq
    
    def getPerpendicularLine(self,point):
        X,Y=point.getXYloc()
        slope=self.getSlope()
        slope=-1/slope
        x,y=sympy.symbols('x y')
        f=(y-Y)-(slope*(x-X))
        eq=sympy.Eq(f,0)
        return eq

    def getFamilyofLines(self,other):
        P=self.getPointOfIntersection(other)
        x1,y1=P.getXYloc()
        x,y,m= sympy.symbols('x y m')
        e=sympy.Eq((y-y1)-m*(x-x1),0)
        return e 
    

class Circle(TwoDegreeEquation):
    def __init__(self, *args):
        """
        Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0

        Takes only 6 args - in order as a,h,b,g,f,c
        """
        if len(args)==6:
            TwoDegreeEquation.__init__(self,*args)
            if self.getShape()!="Circle":
                raise ValueError("Not a Cirlce")
            if self.a!=0:
                self.g/=self.a
                self.f/=self.a
                self.c/=self.a
                self.b/=self.a
                self.a/=self.a
            self.radius= math.sqrt(self.g**2 + self.f**2 -self.c)
        else:
            raise TypeError("Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0 \n\n \
            TwoDegreeEquation takes only 6 args - in order as a,h,b,g,f,c")
        
    def setValue(self, *args):
        self.__init__(*args)

    def __str__(self):
        return str(self.getEquation())

    def getEquation(self):
        """
        Returns the mathematical equation of any 2 degree equation
        """
        y, x = sympy.symbols(('y', 'x'))
        #ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0
        sympy.init_printing()
        return sympy.Eq((self.a*(x**2)) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x)+(2*self.f*y)+ self.c,0)

    def getRadius(self):
        return self.radius

    def getCenter(self):
        return(Point(-self.g,-self.f))

    def draw(self):
        angle = np.linspace( 0 , 2 * np.pi , 300) 
 
        radius = self.radius
        print(radius)
        print(self.g)
        
        x = -self.g+(radius * np.cos( angle ))
        y = -self.f+(radius * np.sin( angle )) 
        
        axes=plt.gca()
        axes.set_aspect( 1 ,adjustable='datalim') 
        
        axes.plot( x, y ) 
         

        #print(plt)
        plt.title( 'Circle' ) 
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        
        plt.show()
    
    def area(self):
        r= self.getRadius()
        return math.pi * r**2

    def resize(self, factor):
        self.radius*=(factor)

    def FamilyOfCircles(self,other):
        x,y,b=sympy.symbols('x y b')
        c1=self.a*(x**2) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x)+(2*self.f*y)+ self.c
        c2=other.a*(x**2) + (2*other.h*x*y) + (other.b*(y**2)) + (2*other.g*x)+(2*other.f*y)+ other.c
        
        exp=(c1+b*c2)
        family= sympy.Eq(exp,0)
        return family

    def getChordLength(self,X,Y):
        """
        X and Y parameters are co-ordinates of the midpoint of the chord
        """
        if self.isPointInside(X,Y):
            Po=self.getCenter()
            r= self.getRadius()
            P1=Point(X,Y)
            d=(distance(Po,P1))
            d=str(d)
            #print(d)
            d=float(d)
            L=2*math.sqrt(r**2- d**2)
            return L


    def isPointInside(self,X,Y):
        F=self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F<0 else False
    def isPointOutside(self,X,Y):
        F=self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F>0 else False
    
    def EquationOfChord(self,X,Y):
        """
        X and Y parameters are co-ordinates of the midpoint of the chord
        """
        y, x = sympy.symbols(('y', 'x'))
        
        S1=self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c
        T= self.a*X*x + self.h*x*Y+self.h*y*X + self.b*Y*y + self.g*(X+x) + self.f*Y*y + self.c
        f=T-S1
        F=sympy.Eq(T-S1,0)
        return F

    def RadicalAxis(self,other):
        x,y =sympy.symbols('x y')
        s1=self.a*x**2 + 2*self.h*x*y + self.b*y**2 + 2*self.g*x + 2*self.f*y + self.c
        s2=other.a*x**2 + 2*other.h*x*y + other.b*y**2 + 2*other.g*x + 2*other.f*y + other.c
        f=s1-s2
        F=sympy.Eq(f)
        return F
    

class Triangle(Shape):
    def __init__(self,l1:StraightLine,l2:StraightLine,l3:StraightLine):
        self.line1=l1
        self.line2=l2
        self.line3=l3
        self.point1=l1.getPointOfIntersection(l2)#A
        self.point2=l2.getPointOfIntersection(l3)#B
        self.point3=l3.getPointOfIntersection(l1)#C
        self.angle1=l1.getAngle(l2)
        self.angle2=l2.getAngle(l3)
        self.angle3=l3.getAngle(l1)
        self.slope1=l1.getSlope()
        self.slope2=l2.getSlope()
        self.slope3=l3.getSlope()
        Shape.__init__(self,"Triangle")
        
        if not self.isTriangle():
            raise ValueError("It is not a traingle")
    
    def setTriangle(self,l1,l2,l3):
        self.__init__(l1,l2,l3)

    def getTriangle(self):
        return self.line1,self.line2,self.line3
    
    def isAcuteAngleTriangle(self):
        if self.angle2<90 and self.angle1<90 and self.angle3<90:
            return True
        return False
    
    def isRightAngleTriangle(self):
        if self.angle2==90 or self.angle1==90 or self.angle3==90:
            return True
        return False
    def isTriangle(self):
        a=self.C=float(str(distance(self.point1,self.point2)))
        b=self.A=float(str(distance(self.point2,self.point3)))
        c=self.B=float(str(distance(self.point3,self.point1)))
        if a+b>=c and b+c>=a and c+a>=b:
            return True
        else:
            return False
    
    def isObtuseAngleTriangle(self):
        if self.angle2>=90 or self.angle1>=90 or self.angle3>=90:
            return True
        return False

    def getTrianglePoints(self):
        return self.point1,self.point2,self.point3
    
    def getLineCoeficients(self):
        l1,l2,l3=self.getTriangle()
        line1=(l1.a,l1.b,l1.c)
        line2=(l2.a,l2.b,l2.c)
        line3=(l3.a,l3.b,l3.c)
        return line1,line2,line3


    def getTriangleAngles(self):
        return self.angle1,self.angle2,self.angle3

    def getTraingleSides(self):
        return self.A,self.B,self.C 

    def getTriangleSlopes(self):
        return self.slope1,self.slope2,self.slope3
    
    def isEquilateralTriangle(self):
        if self.A==self.B==self.C==0 or self.angle3==self.angle2==self.angle1==60:
            return True
        return False

    def isScaleneTraingle(self):
        if self.A!=self.B!=self.C or self.angle3!=self.angle2!=self.angle1:
            return True
        else:
            return False
    def isIsocelesTriangle(self):
        if self.A==self.B or self.B==self.C or self.C==self.A:
            return True
        else:
            return False
    def area(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        a=math.sqrt(s*(s-a)*(s-b)*(s-c))
        return a
    def draw(self):
        l1,l2,l3=self.getTriangle()
        a1,b1,c1= l1.a,l1.b,l1.c
        a2,b2,c2= l2.a,l2.b,l2.c
        a3,b3,c3= l3.a,l3.b,l3.c
        x = np.linspace(-100, 100, 300)
        y = np.linspace(-100, 100, 300)
        X, Y = np.meshgrid(x, y)
        F1 = a1*X+b1*Y+c1
        F2 = a2*X+b2*Y+c2
        F3 = a3*X+b3*Y+c3


        fig,ax = plt.subplots()
        ax.contour(X, Y, F1,levels=[0]) # take level set corresponding to 0
        ax.contour(X, Y, F2,levels=[0])
        ax.contour(X, Y, F3,levels=[0])
        plt.show()


    def getCircumcenterLength(self):
        c=(self.A*self.B*self.C)/(4*self.area())
        return c
    def getInradiusLength(self):
        s=(self.A+self.B+self.C)/2 
        r=self.area()/s 
        return r
    
    def getExCircleRadius(self):
        s=(self.A+self.B+self.C)/2
        a=self.area()
        r1=a/(s-self.A)
        r2=a/(s-self.B)
        r3=a/(s-self.C)
        return r1,r2,r3
    
    def getAngleBisectorLengthPoint1(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        #angle=self.angle1
        cosAby2=math.sqrt((s*(s-a))/(b*c))
        l=(2*b*c*cosAby2)/(b+c)
        return l

    def getAngleBisectorLengthPoint2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        #angle=self.angle1
        cosBby2=math.sqrt((s*(s-b))/(a*c))
        l=(2*a*c*cosBby2)/(a+c)
        return l

    def getAngleBisectorLengthPoint3(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        #angle=self.angle1
        cosCby2=math.sqrt((s*(s-c))/(b*c))
        l=(2*a*b*cosCby2)/(a+b)
        return l
    

    def getMedianLengthPoint1(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        
        l=0.5*math.sqrt(2*(b**2)+2*(c**2)-(a**2))
        return l
    
    def getMedianLengthPoint2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        
        l=0.5*math.sqrt(2*(a**2)+2*(c**2)-(b**2))
        return l
    
    def getMedianLengthPoint3(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        
        l=0.5*math.sqrt(2*(b**2)+2*(a**2)-(c**2))
        return l
    
    def getAltitudePoint1(self):
        a,b,c=self.A,self.B,self.C
        
        A=self.area()
        l=2*A/a
        return l

    def getAltitudePoint2(self):
        a,b,c=self.A,self.B,self.C
        
        A=self.area()
        l=2*A/b
        return l

    def getAltitudePoint3(self):
        a,b,c=self.A,self.B,self.C
        
        A=self.area()
        l=2*A/c
        return l
    
    def getOrthocenterLengthPoint1(self):
        a,b,c=self.A,self.B,self.C
        cosA= ((b**2)+(c**2)-(a**2))/(2*b*c)
        R= self.getCircumcenterLength()
        l=2*R*cosA
        return l

    def getOrthocenterLengthPoint2(self):
        a,b,c=self.A,self.B,self.C
        cosB= ((a**2)+(c**2)-(b**2))/(2*a*c)
        R= self.getCircumcenterLength()
        l=2*R*cosB
        return l

    def getOrthocenterLengthPoint3(self):
        a,b,c=self.A,self.B,self.C
        cosC= ((b**2)+(a**2)-(c**2))/(2*a*b)
        R= self.getCircumcenterLength()
        l=2*R*cosC
        return l

    def sinPoint1(self):
        R=self.getCircumcenterLength()
        sinA= self.A/R
        return sinA

    def sinPoint2(self):
        R=self.getCircumcenterLength()
        sinB= self.B/R
        return sinB

    def sinPoint3(self):
        R=self.getCircumcenterLength()
        sinC= self.C/R
        return sinC

    def sinPoint1by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        sinAby2=math.sqrt(((s-b)*(s-c))/(b*c))
        return sinAby2
    
    def sinPoint2by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        sinBby2=math.sqrt(((s-a)*(s-c))/(a*c))
        return sinBby2
    
    def sinPoint3by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        sinCby2=math.sqrt(((s-b)*(s-a))/(b*a))
        return sinCby2

    
    def cosPoint1(self):
        a,b,c=self.A,self.B,self.C
        cosA= ((b**2)+(c**2)-(a**2))/(2*b*c)
        return cosA

    def cosPoint2(self):
        a,b,c=self.A,self.B,self.C
        cosB= ((a**2)+(c**2)-(b**2))/(2*a*c) 
        return cosB   

    def cosPoint3(self):
        a,b,c=self.A,self.B,self.C
        cosC= ((b**2)+(a**2)-(c**2))/(2*b*a)
        return cosC

    def cosPoint1by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        cosAby2=math.sqrt((s*(s-a))/(b*c))
        return cosAby2

    def cosPoint2by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        cosBby2=math.sqrt((s*(s-b))/(a*c))
        return cosBby2

    def cosPoint3by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        cosCby2=math.sqrt((s*(s-c))/(a*b))
        return cosCby2

    def tanPoint1by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        A=self.area()
        tanAby2=A/(s*(s-a))
        return tanAby2

    
    def tanPoint2by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        A=self.area()
        tanBby2=A/(s*(s-b))
        return tanBby2
    
    
    def tanPoint3by2(self):
        a,b,c=self.A,self.B,self.C
        s=(a+b+c)/2
        A=self.area()
        tanCby2=A/(s*(s-c))
        return tanCby2

    def isPointInsideTriangle(self,point:Point):
        x,y=point.getXYloc()
        l1,l2,l3=self.getTriangle()
        a1,b1,c1= l1.a,l1.b,l1.c
        a2,b2,c2= l2.a,l2.b,l2.c
        a3,b3,c3= l3.a,l3.b,l3.c
        if b1<0:
            a1,b1,c1=-a1,-b1,-c1
        if b2<0:
            a2,b2,c2=-a2,-b2,-c2
        if b3<0:
            a1,b1,c1=-a1,-b1,-c1
        F=(a1*x+b1*y+c1)*(a2*x+b2*y+c2)*(a3*x+b3*y+c3)
        if F<0:
            return True
        else:
            return False
    
    def isPointOutsideTriangle(self,point:Point):
        x,y=point.getXYloc()
        l1,l2,l3=self.getTriangle()
        a1,b1,c1= l1.a,l1.b,l1.c
        a2,b2,c2= l2.a,l2.b,l2.c
        a3,b3,c3= l3.a,l3.b,l3.c
        if b1<0:
            a1,b1,c1=-a1,-b1,-c1
        if b2<0:
            a2,b2,c2=-a2,-b2,-c2
        if b3<0:
            a1,b1,c1=-a1,-b1,-c1
        F=(a1*x+b1*y+c1)*(a2*x+b2*y+c2)*(a3*x+b3*y+c3)
        if F>0:
            return True
        else:
            return False
    
    def getCicumcenter(self):
        p1,p2,p3=self.getTrianglePoints()
        x1,y1=p1.getXYloc()
        x2,y2= p2.getXYloc()
        x3,y3= p3.getXYloc()
        sinA,sinB,sinC=self.sinPoint1(),self.sinPoint2(),self.sinPoint3()
        cosA,cosB,cosC=self.cosPoint1(),self.cosPoint2(),self.cosPoint3()
        sin2A,sin2B,sin2C=2*sinA*cosA,2*sinB*cosB,2*sinC*cosC
        x=((x1*sin2A)+(x2*sin2B)+(x3*sin2C))/(sin2B+sin2A+sin2C)
        y=((y1*sin2A)+(y2*sin2B)+(y3*sin2C))/(sin2B+sin2A+sin2C)
        return Point(x,y)

    def getIncenter(self):
        p1,p2,p3=self.getTrianglePoints()
        x1,y1=p1.getXYloc()
        x2,y2= p2.getXYloc()
        x3,y3= p3.getXYloc()
        a,b,c=self.A,self.B,self.C
        x=(a*x1+b*x2+c*x3)/(a+b+c)
        y=(a*y1+b*y2+c*y3)/(a+b+c)
        return Point(x,y)

    def getExcenters(self):
        #Triangle ABC is the pedal triangle of Triangle I1I2I3
        p1,p2,p3=self.getTrianglePoints()
        x1,y1=p1.getXYloc()
        x2,y2= p2.getXYloc()
        x3,y3= p3.getXYloc()
        a,b,c=self.A,self.B,self.C
        ix1=(-a*x1+b*x2+c*x3)/(-a+b+c)
        iy1=(-a*y1+b*y2+c*y3)/(-a+b+c)
        ix2=(a*x1-b*x2+c*x3)/(a-b+c)
        iy2=(a*y1-b*y2+c*y3)/(a-b+c)
        ix3=(a*x1+b*x2-c*x3)/(a+b-c)
        iy3=(a*y1+b*y2-c*y3)/(a+b-c)
        return Point(ix1,iy1),Point(ix2,iy2),Point(ix3,iy3) #I1,I2,I3
    
    def getSidesOfExcenteralTriangle(self):
        R=self.getCircumcenterLength()
        A=4*R*self.cosPoint1by2()
        B=4*R*self.cosPoint2by2()
        C=4*R*self.cosPoint3by2()

        return A,B,C
    
    def getAnglesOfExcentralTriangle(self):
        A,B,C= self.getTriangleAngles()
        a=90-(A/2)
        b=90-(B/2)
        c=90-(C/2)
        return a,b,c

    def getOrthocenterOfExcentralTriangle(self):
        p=self.getIncenter()
        return p
    

    
    def getOrthocenter(self):
        p1,p2,p3=self.getTrianglePoints()
        x1,y1=p1.getXYloc()
        x2,y2= p2.getXYloc()
        x3,y3= p3.getXYloc()
        tanA2,tanB2,tanC2=self.tanPoint1by2(),self.tanPoint2by2(),self.tanPoint3by2()
        tanA=(2*tanA2)/(1-(tanA2**2))
        tanB=(2*tanB2)/(1-(tanB2**2))
        tanC=(2*tanC2)/(1-(tanC2**2))

        x=((x1*tanA)+(x2*tanB)+(x3*tanC))/(tanB+tanA+tanC)
        y=((y1*tanA)+(y2*tanB)+(y3*tanC))/(tanB+tanA+tanC)
        return Point(x,y)



    def getCentroid(self):
        p1,p2,p3=self.getTrianglePoints()
        x1,y1=p1.getXYloc()
        x2,y2= p2.getXYloc()
        x3,y3= p3.getXYloc()
        x=(x1+x2+x3)/3
        y=(y1+y2+y3)/3
        return Point(x,y)
    
    def DistanceBetweenCircumcenterOrthocenter(self):
        Pc=self.getCicumcenter()
        Po=self.getOrthocenter()
        d=float(str(distance(Pc,Po)))
        return d

    def DistanceBetweenCircumcenterIncenter(self):
        Pc=self.getCicumcenter()
        Pi=self.getIncenter()
        d=float(str(distance(Pc,Pi)))
        return d

    def DistanceBetweenCircumcenterCentroid(self):
        Pc=self.getCicumcenter()
        Pg=self.getCentroid()
        d=float(str(distance(Pc,Pg)))
        return d

    def DistanceBetweenIncenterOrthocenter(self):
        Pi=self.getIncenter()
        Po=self.getOrthocenter()
        d=float(str(distance(Pi,Po)))
        return d

    def DistanceBetweenCentroidOrthocenter(self):
        Pg=self.getCentroid()
        Po=self.getOrthocenter()
        d=float(str(distance(Pg,Po)))
        return d


class Quadilateral(Shape):
    def __init__(self,l1:StraightLine,l2:StraightLine,l3:StraightLine,l4:StraightLine):
        
        self.line1=l1
        self.line2=l2
        self.line3=l3
        self.line4=l4

        self.point1=l1.getPointOfIntersection(l2)#A
        self.point2=l2.getPointOfIntersection(l3)#B
        self.point3=l3.getPointOfIntersection(l4)#C
        self.point4=l4.getPointOfIntersection(l1)#D

        self.side1= float(str(distance(self.point1,self.point2)))
        self.side2= float(str(distance(self.point2,self.point3)))
        self.side3= float(str(distance(self.point3,self.point4)))
        self.side4= float(str(distance(self.point4,self.point1)))

        self.angle1=l1.getAngle(l2)
        self.angle2=l2.getAngle(l3)
        self.angle3=l3.getAngle(l4)
        self.angle4=l4.getAngle(l1)
        
        self.slope1=l1.getSlope()
        self.slope2=l2.getSlope()
        self.slope3=l3.getSlope()
        self.slope4=l4.getSlope()

        self.shape=self.checkShape()
        Shape.__init__(self,self.shape)

    def setQuadilateral(self,l1,l2,l3,l4):
        self.__init__(l1,l2,l3,l4)

    def __str__(self):
        return str(self.getLineEquations())

    def getQuadilateral(self):
        return self.line1,self.line2,self.line3,self.line4
    
    def getSideLength(self):
        return self.side1,self.side2,self.side3,self.side4
    
    def getSlopes(self):
        return self.slope1,self.slope2,self.slope3,self.slope4
    
    def getAngles(self):
        return self.angle1,self.angle2,self.angle3,self.angle4
    
    def getPoints(self):
        return self.point1,self.point2,self.point3,self.point4
    
    def getCenter(self):
        #raise NotImplementedError("getCenter() not implemented for the object")
        #a=self.point1
        #b=self.point3
        #c=self.point1.findMidpoint(self.point3)
        #return c
        x,y=sympy.symbols('x y')
        d1,d2=self.getDiagonalEquations()
        sol=sympy.solve((d1,d2),(x,y))
        return Point(sol[x],sol[y])


    

    def draw(self):
        l1,l2,l3,l4=self.getSquare()
        a1,b1,c1= l1.a,l1.b,l1.c
        a2,b2,c2= l2.a,l2.b,l2.c
        a3,b3,c3= l3.a,l3.b,l3.c
        a4,b4,c4= l4.a,l4.b,l4.c

        x = np.linspace(-100, 100, 300)
        y = np.linspace(-100, 100, 300)
        X, Y = np.meshgrid(x, y)
        F1 = a1*X+b1*Y+c1
        F2 = a2*X+b2*Y+c2
        F3 = a3*X+b3*Y+c3
        F4 = a4*X+b4*Y+c4

        fig,ax = plt.subplots()
        ax.contour(X, Y, F1,levels=[0]) # take level set corresponding to 0
        ax.contour(X, Y, F2,levels=[0])
        ax.contour(X, Y, F3,levels=[0])
        ax.contour(X, Y, F4,levels=[0])
        plt.show()

    def area(self):
        raise NotImplementedError("area() not implemented for the object")
    
    def getDiagonalLength(self):
        a=self.point1
        b=self.point2
        c=self.point3
        d=self.point4
        d1= float(str(distance(a,c)))
        d2= float(str(distance(b,d)))
        return d1,d2

    def getDiagonalEquations(self):
        A,B,C,D=self.getPoints()
        ax1,ay1=A.getXYloc()
        bx1,by1=B.getXYloc()
        cx1,cy1=C.getXYloc()
        dx1,dy1=D.getXYloc()
        m1=(cy1-ay1)/(cx1-ax1)
        m2=(by1-dy1)/(bx1-dx1)
        x,y=sympy.symbols('x y')
        d1=sympy.Eq((y-ay1)-m1*(x-ax1),0)
        d2=sympy.Eq((y-by1)-m2*(x-bx1),0)
        return d1,d2
        
    
    def getLineEquations(self):
        return self.line1,self.line2,self.line3,self.line4
    
    def getPerimeter(self):
        a,b,c,d=self.getSideLength()
        return a+b+c+d

    def checkShape(self):
        a,b,c,d=self.getSideLength()
        A,B,C,D=self.getAngles()
        #p1,p2,p3,p4=self.getPoints()
        m1,m2,m3,m4=self.getSlopes()

        if a==b==c==d and A==B==C==D==90 and m1==m3 and m2==m4:
            return "Square"
        elif a==b==c==d and A==C and B==D and A!=B and m1==m3 and m2==m4:
            return "Rhombus"
        elif a==c and b==d and a!=b and A==B==C==D==90 and m1==m3 and m2==m4:
            return "Rectangle"
        elif a==c and b==d and a!=b and A==C and B==D and A!=B and m1==m3 and m2==m4:
            return "Parallelogram"
        elif ((m1==m3 and m2!=m4) or(m1!=m3 and m2==m4)) and m1 !=m2 and a!=c:
            return "Trapezium"
        elif A+C==180 and B+D==180:
            return "CyclicQuadilateral"
        elif a!=b!=c!=d:
            return "Quadilateral"

    def getAngleBisectors(self):
        raise NotImplementedError("method not implemented")

    def getDiagonalAngles(self):
        d1,d2=self.getDiagonalEquations()
        angle1= d1.getAngle(d2)
        angle2= 180-angle1
        return angle1,angle2

    

        

class Square(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        Quadilateral.__init__(self,l1, l2, l3, l4)
        if self.shape != "Square":
            raise ValueError("The given lines do not form a Square")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c
    
    def area(self):
        a,b,c,d=self.getSideLength()
        return a*b

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()
    
    def getInradiusLength(self):
        a=self.getDiagonalLength()
        r=a/(2*math.sqrt(2))
        return r

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    
    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()
        def area(x1, y1, x2, y2, x3, y3):
      
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()


class Rectangle(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        super().__init__(l1, l2, l3, l4)
        if self.shape != "Rectangle":
            raise ValueError("The given lines do not form a Rectangle")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    def area(self):
        a,b,c,d=self.getSideLength()
        return a*b

    def isGoldenRectangle(self):
        a,b=self.getSideLength()
        l= a*b + b*b
        k=a*a
        if l==k:
            return True
        return False

    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()
        def area(x1, y1, x2, y2, x3, y3):
      
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()

        
class Rhombus(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        super().__init__(l1, l2, l3, l4)
        if self.shape != "Rectangle":
            raise ValueError("The given lines do not form a Rectangle")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    def area(self):
        d1,d2=self.getDiagonalLength()
        return d1*d2
    
    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()
        def area(x1, y1, x2, y2, x3, y3):
      
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()
    

class Parallelogram(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        super().__init__(l1, l2, l3, l4)
        if self.shape != "Rectangle":
            raise ValueError("The given lines do not form a Rectangle")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    def area(self):
        d1,d2=self.getDiagonalLength()
        a1,a2=self.getDiagonalAngles()
        r=(math.pi*a1)/180
        return d1*d2* math.sin(r)
    
    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()
        def area(x1, y1, x2, y2, x3, y3):
      
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()

class Parabola(TwoDegreeEquation):
    def __init__(self, *args):
        """
        Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0

        Takes only 6 args - in order as a,h,b,g,f,c
        """
        if len(args)==6:
            TwoDegreeEquation.__init__(self,*args)
            if self.getShape()!="Parabola":
                raise ValueError("Not a Parabola")
            
        else:
            raise TypeError("Consider the general two degree equation as ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0 \n\n \
            TwoDegreeEquation takes only 6 args - in order as a,h,b,g,f,c")

    def setValue(self, *args):
        self.__init__(*args)

    def __str__(self):
        return str(self.getEquation())

    def getEquation(self):
        """
        Returns the mathematical equation of any 2 degree equation
        """
        y, x = sympy.symbols(('y', 'x'))
        #ax^2 + 2hxy + by^2 + 2gx + 2fy +c=0
        sympy.init_printing()
        return sympy.Eq((self.a*(x**2)) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x)+(2*self.f*y)+ self.c,0)
    
    def isPointInside(self,X,Y):
        F=self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F<0 else False

    def isPointOutside(self,X,Y):
        F=self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F>0 else False

    def draw(self):
        x = np.linspace(-1000, 1000, 3000)
        y = np.linspace(-1000, 1000, 3000)
        X, Y = np.meshgrid(x, y)
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c

        fig,ax = plt.subplots()
        ax.contour(X, Y, F, levels=[0]) # take level set corresponding to 0
        plt.show()
    
    def getFocus():
        pass

    def getVertex():
        pass
    
    def getEccentricity(self):
        return 1.0


        


        
    
    
    
    
    

        




    








    
















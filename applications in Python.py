import numpy as np
mylist =[[1,2,3],[3,4,5]]
myarray = np.array(mylist)
print(("matrix:%s")%str(myarray))
print(("matrix dimensionis : %s")% str(myarray.shape))
print((f"first row is {myarray[0]}"))
print((f"last row is {myarray[-1]}"))
print((f"first row third colum is {myarray[0,2]}"))
print((f"third colum is {myarray[: ,2]}"))# كل الصفوف بس هات العمود التالت بس 
###############################################################################
months = ["jan","feb","mar",
           "apr","may","jun","JUl",
           "auq","sep","oct","nov","dec"]
sun = [44.7,65.5,101.7,148.3,170.9,171.4,176.7,
        186.1,133.9,105.4,59.6,45.8]
for s , m in sorted(zip(sun,months), reverse=True):
    print(f"{m}:{s:5.1f} hrs")
############################ لعبة القسمه ###################################################
FB = []
for t in range(21):
    if t%5 == 0 and t%3 ==0 :
        FB.append("fizzbuzz")
    elif t%3 ==0 and t%5 !=0 :
        FB.append("fizz")
    elif t%5 ==0 and t%3 !=0 :
         FB.append("buzz")
    else:
        FB.append(t)
print(FB)
##################################اثبات ال Pi#############################################
import math
pi = 0
for k in range(20):
    pi += pow(-3,-k) /(2*k+1)
pi *= math.sqrt(12)
print("pi =",pi)
print("error =", abs(pi - math.pi))
#####################################################################################
# دالة ال sin
import math
import pylab
Xmin,Xmax =-2. *math.pi ,2. *math.pi
n = 1000 
x =[0.] *n
y =[0.] *n
dx = (Xmax-Xmin)/(n-1)
for i in range (n):
    xpt = Xmin + i * dx
    x[i]= xpt
    y[i]=math.sin(xpt)**2
    
pylab.plot(x, y)
####################################################################
import pylab
years = range(2000,2010)
divorce_rate = [5.0,4.7,4.6,4.5,4.5,4.4,4.3,4.2,4.2,4.1]
margaine_consumption = [8.2,7,6.5,5.3,5.2,4,4.6,4.5,4.2,3.7]

line1= pylab.plot(years, divorce_rate,"b-o",label="divorce rate im maine")
pylab.ylabel(" divorces per 1000 people")
pylab.legend()
pylab.twinx()
line2= pylab.plot(years,margaine_consumption,"r-o",label= "margarine cons")
pylab.ylabel("1b of margarine(per capita)")
lines = line1 + line2
labels = []
for line in lines :
    labels.append(line.get_label())
pylab.legend(lines,labels)
pylab.show()
################## رسم بياني ###############
import matplotlib.pyplot as plt 

years = range(2000, 2010)
divorce_rate = [5.0, 4.7, 4.6, 4.5, 4.5, 4.4, 4.3, 4.2, 4.2, 4.1]
margarine_consumption = [8.2, 7, 6.5, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]

plt.figure(figsize=(16, 6))
line1, = plt.plot(years, divorce_rate, 'b-o', label='Divorce Rate in Maine')
plt.xlabel('Year')
plt.ylabel('Divorces per 1000 people')
plt.twinx()
line2, = plt.plot(years, margarine_consumption, 'r-o', label='Margarine Consumption')
plt.ylabel('1 lb of margarine (per capita)')

plt.title('Divorce Rate and Margarine Consumption in Maine (2000-2009)')

# عرض العلامات عن طريق إنشاء مربع العلامة
plt.legend(handles=[line1, line2], loc='upper right')

plt.show()
#######################################حساب كثافة الكواكب #########################
import math
body = {"sun" :(1.988e30,6.955e5),
        "mercury" :(3.301e23,2440),
        "venus" : (4.867e+24,6052),
        "earth": (5.972e24,6371.),
        "mars" : (6.417e23,3390.),
        "jupiter":(1.899e27,69911.),
        "saturn" :(5.685e26,58232.),
        "urauns" :(8.682e25, 25362.),
        "neptune": (1.024e26,24622.)
        }
planets = list(body.keys())
planets.remove("sun")
def calc_density(m,r):
    """return the density of a sphere with mass m and radius r . """
    return m / (4/3 * math.pi * r**3)
rho = {}
for planet in planets:
    m ,r=body[planet]
    rho[planet] =  calc_density(m*1000, r*1.e5)
for planet,density in sorted(rho.items()):
  print("the density of {0} is {1:3.2f} g/cm3".format(planet , density)) 
############################### مقلوب المصفوفه #####################
def makeT(mm):
    rr =len(mm)
    cc =len(mm[0])
    tm = [[0 for i in range(rr)] for j in range(cc)]
    for ccc in range(cc):
        for rrr in range(cc):
            tm[ccc][rrr] = m[rrr][ccc]
    return tm
m = [[1,20,3,4],
     [4,5,61,8],
     [7,8,9,60],
     [30,3,6,3],
     [0,7,60,9]]
print(makeT(m))
######################## best fit line #######################
import pylab

def lreg(xx, yy):
    xdash = pylab.mean(xx)
    ydash = pylab.mean(yy)
    z = []
    w = []

    for g in range(len(xx)):
        z.append(float(xx[g]) * float(yy[g]))
        w.append(float(xx[g]) * float(xx[g]))

    xydash = pylab.mean(z)
    x2dash = pylab.mean(w)

    m = (xydash - (xdash * ydash)) / (x2dash - (xdash ** 2))
    c = ydash - (m * xdash)

    return round(m, 5), round(c, 5)

xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ydata = [4.8, 5.0, 7.5, 7.5, 9.4, 9.5, 11.7, 12.0, 11.2, 15.0]

mm, cc = lreg(xdata, ydata)
print(f"Slope is {mm}")
print(f"C value is {cc}")

exactx = []
exacty = []

for i in range(1, 11):
    exactx.append(i)
    exacty.append(cc + (i * mm))

pylab.plot(xdata, ydata, "o", markersize=6, color="b")
pylab.plot(exactx, exacty, linewidth=2, color="r")
pylab.show()
####################best fit line##############################

import numpy as np
import pylab

Polynomial = np.polynomial.Polynomial

conc = np.array([0, 20, 40, 80, 120, 180, 260, 400, 800, 1500])
A = np.array([2.287, 3.528, 4.336, 6.909, 8.274, 12.855, 16.085, 24.797, 49.058, 89.400])

cmin, cmax = min(conc), max(conc)
pfit, stats = Polynomial.fit(conc, A, 1, full=True, domain=(cmin, cmax))

print("Raw fit results:", pfit, stats)

A0, m = pfit
resid, rank, sing_val, rcond = stats
rms = np.sqrt(resid[0] / len(A))

print(f"Fit: A = {m:.3f} * c + {A0:.3f}")
print(f"RMS residual = {rms:.4f}")

pylab.plot(conc, A, 'o', color="r")
pylab.plot(conc, pfit(conc), color="b")
pylab.show()
############################## الارقام الأوليه ###########################
def gen_primes(N):
    primes = set()
    for n in range(2, N):
        if all(n % p > 0 for p in primes):
            primes.add(n)
            yield n

print(*gen_primes(100))
############################## الارقام الأوليه ###########################
def pn(n):
    pnn = []
    for h in range(2, n + 1):
        divisible = True
        for j in range(2, h):
            if (h % j) == 0:
                divisible = False
                break
        if divisible:
            pnn.append(h)
    return pnn

n = 100  # يمكنك استبدال 100 بالرقم الذي ترغب فيه
prime_numbers = pn(n)
print(*prime_numbers)
############################رسم 3d ################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

L = 2
n = 400
x = np.linspace(-L, L, n)
y = x.copy()

X, Y = np.meshgrid(x, y) #   بيخلي xو ال y زي شبكه  متداخله 
Z = np.exp(-(X**2 + Y**2))

fig, ax = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': '3d'})

ax[0, 0].plot_wireframe(X, Y, Z, rstride=40, cstride=40)
ax[0, 1].plot_surface(X, Y, Z, rstride=40, cstride=40, cmap=cm.jet)
ax[1, 0].plot_surface(X, Y, Z, rstride=12, cstride=12, cmap=cm.jet)
ax[1, 1].plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.hot)

for axes in ax.flatten():
    axes.set_xticks([-2, -1, 0, 1, 2])
    axes.set_yticks([-2, -1, 0, 1, 2])
    axes.set_zticks([0, 0.5, 1])

fig.tight_layout() 
# ستساعد في تحسين ترتيب ومظهر الرسم البياني 
 # وظيفة تستخدم لضبط البعد النسبي بين العناصر
plt.show()
############################# scatter##########################################
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

countries = ['Brazil', 'Madagascar', 'S. Korea', 'United States', 'Ethiopia', 'Pakistan', 'China', 'Belize']

birth_rate = [14.25, 33.55, 9.5, 14.25, 38.6, 30.2, 13.5, 23.0]
life_expectancy = [73.7, 64.3, 81.3, 78.8, 63.0, 66.4, 75.2, 73.7]
GDP = np.array([4800, 240, 16700, 37700, 230, 670, 2640, 3490])

fig = plt.figure()
ax = fig.add_subplot(111)

# Some random colors:
colors = range(len(countries))

ax.scatter(birth_rate, life_expectancy, c=colors, s=GDP * 0.1)
fig.tight_layout() 
plt.show()
#(X) وتوقعات العمر
#(y) معدلات الولادة
# حجم الدائره هو حجم الاقتصاد ف الدوله ال GDP
##############################################################################
  
 


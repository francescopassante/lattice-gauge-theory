## **1\. The $Z\_2$ Group Basics**

* **Elements:** $\\{1, \-1\\}$ (also called Ising variables).  
* **Operation:** Multiplication ($1 \\cdot 1 \= 1, \-1 \\cdot \-1 \= 1, 1 \\cdot \-1 \= \-1$).  
* **Nature:** This is the simplest non-trivial gauge group, representing a binary parity/flip symmetry.

## **2\. Lattice Geometry & Field Representation**

* **Sites vs. Links:** Gauge fields live on the **links** (edges) of the lattice, not the sites.  
* **Link Variable ($U\_\\mu(n)$):** In 2D, each site $n$ has two links: one pointing in the $x$ direction ($\\mu=1$) and one in the $y$ direction ($\\mu=2$).  
* **Input Tensor Shape:** For an $L \\times L$ lattice, your NN input is typically a tensor of shape (Batch, 2, L, L).  
* **Boundary Conditions:** We almost always use **Periodic Boundary Conditions (PBCs)** to maintain translational invariance and avoid edge effects.

## **3\. Gauge Symmetry**

* **Local Gauge Transformation:** Defined by a value $\\Omega(n) \\in \\{1, \-1\\}$ at every **site**.  
* **Transformation Law:**  
  $$U'\_\\mu(n) \= \\Omega(n) U\_\\mu(n) \\Omega(n+\\hat{\\mu})$$  
  *Note: A link flips if the transformation values at its two endpoints are different.*

## **4\. Observables & The Action**

* **The Plaquette ($P$):** The smallest gauge-invariant loop. For a square on the lattice:  
  $$P \= U\_1 U\_2 U\_3 U\_4$$  
* **Wilson Action ($S$):** The "energy" of a configuration, used to weight the probability of states ($e^{-S}$).  
  $$ S \= \-\\beta \\sum\_{p} P\_p $$  
  Where $\\beta$ is the coupling constant (inverse temperature).

## **5\. Neural Network Implementation**

* **The Goal:** Build a **Gauge Invariant** network where $f(U) \= f(U')$.  
* **Strategy:**  
  1. Convert links $U$ into a grid of Plaquettes $P$.  
  2. Plaquettes are invariant by definition; any CNN acting on them will be gauge-invariant.  
* **First Test:** Train the network to predict the **Action** $S$.  
* **Verification:** Apply a random gauge transformation to an input configuration. If the network is correctly designed, the output should not change.

---

**6\. Physics & Dimensionality (Q\&A Summary)**

### **Does 2D mean 1 Space \+ 1 Time?**

**Yes.** In Lattice Gauge Theory, we work in **Euclidean Spacetime**. We treat the time dimension as a spatial one.

* 2D \= 1 Space \+ 1 Time.  
* 4D \= 3 Space \+ 1 Time.

### **What is a Phase Transition?**

It is a sharp change in the behavior of the vacuum as you vary $\\beta$.

* **Ordered Phase (High $\\beta$):** Links align, $\\langle P \\rangle \\approx 1$.  
* **Disordered Phase (Low $\\beta$):** Links are random, $\\langle P \\rangle \\approx 0$.  
* **Detection:** Look for a "kink" or jump in the plot of Average Plaquette vs. $\\beta$.

### **Group ($SU(3)$) vs. Dimension**

These are independent concepts:

* **The Dimension ($d$):** Defines the "connectivity" of the grid (how many links per site).  
* **The Group ($G$):** Defines the "math" living on the link (a bit for $Z\_2$, a complex matrix for $SU(3)$).  
* **Why it matters:** While you can put any group in any dimension, the **physics** (like confinement) usually only becomes interesting in 3D or 4D.


### **Data Generation Strategy**

1. **Haar Random:** Assign $\\pm 1$ randomly to links. Good for testing invariance.  
2. **Monte Carlo (Metropolis):** Sample configurations based on $e^{-S}$. Necessary for learning real physics and phase behaviors.

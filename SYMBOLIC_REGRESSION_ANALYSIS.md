# Analysis: Current Symbolic Regression vs Advanced Alternatives

## 🎯 **Current Implementation: Hard-Coded Candidate Models**

The current system uses **19 pre-defined expressions** that are manually crafted based on physics knowledge:

### The 19 Hard-Coded Models:
1. **Linear TF**: `a + b * log_v`
2. **Linear TF (obs)**: `a + b * log_v_obs` 
3. **Linear TF (norm)**: `a + b * v_norm`
4. **Classic TF**: `a - b * log_v`
5. **Classic TF (obs)**: `a - b * log_v_obs`
6. **Classic TF (norm)**: `a - b * v_norm`
7. **Power law TF**: `a + b * log_v * log_v`
8. **Linear power TF**: `a * v_norm`
9. **Quadratic TF**: `a + b * v_norm + c * v_norm²`
10. **TF + stellar mass**: `a - b * log_v + c * log_M_star`
11. **TF + distance**: `a - b * log_v + c * log_D`
12. **TF + velocity²**: `a - b * log_v + c * log_v²`
13. **TF + mass + distance**: `a - b * log_v + c * log_M_star + d * log_D`
14. **TF + mass + velocity²**: `a - b * log_v + c * log_M_star + d * log_v²`
15. **Full systematic**: `a - b * log_v + c * log_M_star + d * log_D + e * log_v²`
16. **TF + quadratic velocity**: `a - b * log_v + c * log_v²`
17. **TF + velocity-mass interaction**: `a - b * log_v + c * log_v * log_M_star`
18. **Normalized TF + mass**: `a - b * v_norm + c * M_star_norm`
19. **Normalized TF + distance**: `a - b * v_norm + c * D_norm`

## ⚙️ **Current Approach: Template-Based "Symbolic Regression"**

The current system is NOT true symbolic regression - it's more like:
- **Template matching**: Pre-defined expression templates
- **Parameter fitting**: Using least-squares optimization 
- **Model selection**: Ranking based on R² and complexity
- **Physics-informed**: Hand-crafted based on known Tully-Fisher physics

### Strengths:
✅ Fast and deterministic
✅ Physics-informed expressions
✅ Easy to interpret results
✅ Focused on astrophysics problem

### Limitations:
❌ Cannot discover novel functional forms
❌ Limited to pre-conceived ideas
❌ Not truly "AI-driven discovery"
❌ May miss unexpected physics

## 🚀 **Advanced Symbolic Regression Alternatives**

### 1. **PySR (Python Symbolic Regression)**
- **Genetic programming** with Julia backend
- **Discovers arbitrary expressions** automatically
- **Pareto-optimal** complexity vs accuracy
- **Highly customizable** operators and constraints

### 2. **AI Feynman**
- **Physics-inspired** symbolic regression
- **Dimensional analysis** built-in
- **Multi-step discovery** process
- **Designed for physical laws**

### 3. **gplearn**
- **Scikit-learn compatible**
- **Genetic programming** in pure Python
- **Custom fitness functions**
- **Tree-based expressions**

### 4. **Operon**
- **High-performance** C++ backend
- **Modern GP algorithms**
- **Multi-objective optimization**
- **Scalable to large datasets**

## 🔬 **Why Use Advanced Methods?**

### **Potential Discoveries Beyond Hard-Coded Models:**
- **Non-polynomial relationships**: e.g., `M = a * sin(b * log(v)) + c * exp(d * log(M_star))`
- **Multi-variable interactions**: Complex dependencies between mass, velocity, distance
- **Emergent physics**: Relationships not anticipated by human intuition
- **Robust to outliers**: GP can evolve expressions that handle edge cases

### **Example Novel Forms PySR Might Discover:**
```python
# Potentially discoverable by PySR but not current system:
"a * log(v * M_star^0.3) + b * sqrt(D) - c * exp(-v/1000)"
"a - b * log(v + c * M_star) + d * sin(0.1 * D)"
"(a * v^b + c) / (d * M_star^e + f)"
```

## 🎯 **Recommendation: Hybrid Approach**

### **Phase 1: Keep Current System** (✅ Already working)
- Validates known physics
- Provides baseline performance
- Fast and interpretable

### **Phase 2: Add Advanced SR** (🚀 Next step)
- **PySR** for general discovery
- **AI Feynman** for physics-aware discovery
- **Comparative analysis** of discoveries

### **Phase 3: Physics Validation** (🔬 Future)
- **Cross-validation** with real observations
- **Physical interpretation** of novel forms
- **Astrophysical significance** testing

## 📊 **Implementation Strategy**

1. **Install PySR**: `pip install pysr`
2. **Run parallel analysis**: Current system + PySR
3. **Compare discoveries**: Hard-coded vs evolved expressions
4. **Physics validation**: Interpret novel forms
5. **Performance comparison**: Accuracy, interpretability, robustness

This would transform the project from "template-based fitting" to **true AI-driven physics discovery**!

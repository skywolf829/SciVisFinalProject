# SciVis final project

An extension of a GitHub repo called Py Stable Fluids, a minimal Stable Fluids inspired fluid solver with Python and NumPy.

In this extension, I've made a few use cases of the solver.
The first is similar to the original code's example, which creates a number of sources in a circle facing the center. I've augmented mine to produce from the source from the start frame until the end frame. I've also added boundary constraints such that velocity on the walls is zero. Lastly, I add a text boundary in the center of the domain. After advecting dye for some period, the letters "SCIVIS" become clear. The visualization is a product of the curl of the flow field and the dye location.

In the second use case, an image is loaded as the dye starting color in the domain, and that is advected with a single sources. Again, boundary constraints are added, so color cannot leave the domain. 

Paper for Stable Fluids: https://d2f99xq7vri1nk.cloudfront.net/legacy_app_files/pdf/ns.pdf

Great video explaining Stable Fluids: https://www.youtube.com/watch?v=766obijdpuU

Original code: https://github.com/GregTJ/stable-fluids

Helpful Reddit post: https://www.reddit.com/r/Python/comments/fkk7aa/fluid_simulation_in_python/fktzp8v/


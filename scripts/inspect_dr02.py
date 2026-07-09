import mujoco
from pathlib import Path


xml_path = Path("assets/robots/dr02/dr02.xml")
model = mujoco.MjModel.from_xml_path(str(xml_path))

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)
print("nbody =", model.nbody)
print("ngeom =", model.ngeom)

print("\n=== Bodies ===")
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

print("\n=== Joints / DoFs ===")
for i in range(model.nv):
    jid = model.dof_jntid[i]
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))

print("\n=== Actuators ===")
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

print("\n=== Sites ===")
for i in range(model.nsite):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))

print("\n=== Geoms ===")
for i in range(model.ngeom):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))

### Implicit Compliance 
## Optional Lab 1 for CS 123, Spring 2025
This optiional lab is modified from the EECS 381 final project from Ankush, Guise, and JC. The goal is to verify if Pupper's legs can infer tactile information without any tactile sensors. No guarentee to work yet!

## Clone repo to Pupper
`cd ~`
`git clone https://github.com/JummerCloth/implicit_compliance.git`
## Pipeline on Pupper
# Enter directory: 
`cd ~/implicit_compliance`

Each of the following jobs would require a separate terminal in `~/implicit_compliance` to run in chronological order:
# Start launch file: 
`ros2 launch ic.launch.py`

# Run Contact Detector 
The current implementation requires a DenseTact sensor (which is very hard to make for regular folks and even more painful to calibrate). You will need to modify this file to make the Pupper's leg compliant without : 
`python3 dt_contact_pub_lw.py`

# Run Compliant Walking Motion
`python3 leg_motion_on_contact`

Log Data (You can modify this to collect more data yourself): `python3 data_collector.py --index experiment<index>`
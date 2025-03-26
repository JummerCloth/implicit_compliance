# Optional Lab 1 for CS 123, Spring 2025
## Modified from the EECS 381 final project from Ankush, Guise, and JC
## Implicit Compliance

### Clone repo to Pupper
`git clone https://github.com/JummerCloth/implicit_compliance.git
### Collect Data
Enter directory: `cd ~/implicit_compliance`

Start launch file: `ros2 launch ic.launch.py`

Run DenseTact (You will need to modify this to run implicit compliance): `python3 dt_contact_pub_lw.py`

Run Walking Motion: `python3 leg_motion_on_contact`

Log Data (You won't need this for this lab): `python3 data_collector.py --index experiment<index>`

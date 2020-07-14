# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

## Implementation
A PID controller was implemented using the same steps given in the lesson and applied on the quizzes. The crosstrack error was inputted into a function where the P, I and D errors were updated. In another function called TotalError, these errors where multiplied to their respective Tau values. The output of this function was then inputted into the car as our steering torque.

## Reflections

Effect of PID values:

# Proportional Gain
The P gain is "proportional" to the error. This value effects how quickly your current state reaches the desired state. If the P value is too high, your desired state will be reached quickly with high overshoot. If your P value is too low, your desired state will be reached slowly with lower overshoot. Just P control is not a good method, so it is always combine as PD or PID (rarely PI).

# Integral Gain
The I gain is how the vehicle takes past errors into account for the new error. Increasing it typically smoothens out the response. Having Integral control in your controller allows your system to minimize error. Without I, error will never be minimized to zero resulting in a control system that never reaches the desired state.

# Derivative Gain
The D gain in our controller determines how much influence the rate of change of error effects the response. Adding D to your controller allows you to predict what the next error will be and reduces overshoot. A high D value can result in jerky movements of a vehicle, while a low D value will result in high overshoot since the system is unable to predict what the next error will be.

## PID Tuning
I did not use a Optimization techinique to find the PID values because it was confusing to apply the twiddle method with this simulator, but that would be some future work I would definitely like to implement. The PID values were tuned manually using the PID values used on the quizzes as a starting point. I started by using P: 0.1, I: 0.004 and D: 0.3 as my PID values but the car overshooted a lot and made sort of like a sine wave pattern. Hence, I increased the D value to 2 and decreased the I value to 0.001. The vehicle performed better but in a tight turn one wheel when off the drivable area. Then, I increased a bit the P gain in order for the vehicle to perform perform quicker on tight turns. This process was repeated multiple times until the vehicle managed to drive through the whole track. It doesn't turn smooth but at least it managed to drive through the track. 

## Simulation
The vehicle successfully drove one full lap by applying a PID controller for the steering and doing some PID tuning. A video was of the vehicle driving around the track was uploaded in a folder called video.

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).


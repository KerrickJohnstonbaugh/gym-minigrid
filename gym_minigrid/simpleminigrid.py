from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class SimpleKey(Key):
    def can_overlap(self):
        """Can the agent overlap with this?"""
        return True

class SimpleGridEnv(MiniGridEnv):
    """
        Simplified mini grid environment with only movement actions
    """
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
    

    def step(self, action):
        '''
        Always try to toggle, pick up
        '''
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        else:
            assert False, "unknown action"

        # Updated the position in front of the agent
        fwd_pos = self.front_pos

        # Update the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Pick up an object
        # Always try
        if fwd_cell and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*fwd_pos, None)       

        # Toggle/activate an object
        # Always Try
        if fwd_cell:
            fwd_cell.toggle(self, fwd_pos)

       



        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}
    
class SimpleDoorKey(SimpleGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeyEnv5x5(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=5)

class DoorKeyEnv6x6(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyEnv16x16(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Simple-DoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5'
)

register(
    id='MiniGrid-Simple-DoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6'
)

register(
    id='MiniGrid-Simple-DoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv'
)

register(
    id='MiniGrid-Simple-DoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv16x16'
)

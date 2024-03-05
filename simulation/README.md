## Initialize the simulation:
1. cd into `shift-rl-sim/simulation`
2. run `simulation_scripts/initial_sim.sh`
3. run `python ppo_parellel.py`
4. After the simulation, remember to go back to SHIFT folder and run `./startup -k`

## Simulation General setup:
1. Open `ppo_parellel.py` located in `/shift_research/simulation` and locate `config_dic`
2. For **Training**:
   - set `resume` to `False`
   - set `seed` to your preferece
   - within `train_args`:
     - set `lr` to `0.0001`
     - adjust `epoch` (note: `epoch1` * `step_per_epoch` = simulation duration)
   - Results:
     - save `~/Results_simulation/iteration_info` and `~/Results_simulation/log/checkpoint`
3. For **Continual Training**:
   - load your saved checkpoint folder into `~/Results_simulation/log/checkpoint`
   - clear `~/Results_simulation/log/event`  ****
   - set `resume` to `True`
   - set `seed` to match the original trained seed
4. For **Testing**:
   - load your saved checkpoint folder into `~/Results_simulation/log/checkpoint`
   - clear `~/Results_simulation/log/event` ****
   - set `resume` to `True`
   - set `seed` to match the original trained seed
   - set `lr` to `0` ****
5. For **Untrained**:
   - set `resume` to `False`
   - set `lr` to `0`
6. For **Flash Agent**:
   - set `is_flash` to `True`
   - Adjust parameters within `flash_rders`:
     - `flash_size`: # of total lots (100 shares) for each window of flash orders
     - `num_orders`: # `flash_size` / `num_orders` = # of lots for each order within the flash window
     - `time_bet_order`: # seconds between each order within the flash window
     - `num_flash`: total # of flash windows
     - `time_bet_flash`: # seconds between each flash window
7. For **Informed LT Agents**:
   - locate `lt_flows` right above `config_dic`
   - `lt_flows`: a list consist 1 buy fraction list and 1 sell fraction list
   - replace `targetBuySellFrac` with `lt_flows` in `config_dic`
   - set `frac_duration`


## Agent Parameters inside `ppo_parellel.py`:
1. Open `ppo_parellel.py` located in `/shift_research/simulation`
2. Locate `config_dic` and adjust the simulation setup
   - `mm`: Market maker
   - `lt`: Liquidity taker
   - `weight`: How much the agents value the PnL component of their reward: [0,1]
   - `gamma`: Risk aversion rate on inventory PnL
   - `tar_m`: Market maker's targeted share of the market: (current agent shares / total shares in LOB): [0,1]
   - `act_sym_range`: Upper and lower bound for market maker's symmetric action
   - `targetBuySellFrac`: Liquidity takers' targeted chance of buying and selling
   - `frac_duration`: duration of the `targetBuySellFrac` in steps, `0` means use the first pair of fraction throughout the simulation
   - `order_size`: # of lots (100 shares) per order placed at each step
3. In `mm_unified_setup`:
   - `rl_t`: # of seconds for each step for market makers
   - `Info_step`: #of seconds for its LOB pulling speed inside another thread
   - `nTimeStep`: # of data its LOB pulling thread saves
   - `ODBK_range`: # of levels of the LOB it saves
   - `max_order_list_length`: #the max of orders it can hold before canceling
4. In `lt_unified_setup`:
   - `step_time`: # of seconds for each step for liquidity takers
   - `normalizer`: a constant applied to the PnL component to reduce its magnitude for reward balancing
   - `order_book_range`: # of levels of the order book given to the RL agent as part of its observations


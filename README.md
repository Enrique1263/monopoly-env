# Monopoly environment for gym library
- Multi Agent supported
- Pygame visualization (4 players currently)
## Hands-on details
To run this for your self, as I haven't registered the library, you'll need to follow these steps
- Clone the repository

```bash
git clone https://github.com/Enrique1263/monopoly-env
```

- Go to /monopoly_env folder and install the library

```bash
cd ./monopoly_env/
```
```bash
pip install .
```

- Now you can use the gym environment in your own python files
```python
import monopoly_env
import gym

env = gym.make('MonopolyEnv-v0', **kwargs)
```
## Project structure
This environment relies on three pillars:
- The gym.env class
- The pygame board class
- The player class (It's a suggestion, although in the future could be mandatory to inherit from it)
## Monopoly rules and considerations
For it's current state, this game only considers buying a property or a house, being the second one only when you land on the property. Stations and public services' logic it's implemented, although at this moment you can't buy neither of them. Community chest and chance logic it's semi-implemented, that's why it's not being used.

Other rules are, going to jail if you throw doubles thrice, taxes and card debts go to free parking and the player who lands get's it all, no need for color group to edificate (may be the other way in an advanced state of the environment).

If you are a fan of monopoly you'd known there are dozens of versions, that's why I made it so you can custom your own game. For names you'll have an txt of a dictionary tile:name, for cards you'll have two txt of list text|money|afects others?|tile (as it's not being use it's not definitive), for more details take a look at mines in the cards folder.
## Future of this project
I'm not sure of what of the things from bellow would be done, or if I am going to continue with this project, but to whom it may concern this is my roadmap:
- Finishing the stations and public services
- Making the visualization up to four instead of only four
- Having an option for the visualization being more than four (tokens wouldn't be very visible, but my intention is that 8 people could distinguis there tokens)
- Enhancing the visualization to make it more pretty and explanatory.
- Adding community chest and chance cards logic
- Buying a house from where ever
- Needing group color to buy a house (optional rule is my intention)

Now there's a big wall that means a lot of interaction, whether is the human or the agent
- Selling or mortaging houses/properties in your turn
- Bargaining with other player in your turn
- Doing it when it's not your turn

\
What i have in mind for this last approach is having an action array for each possible consecutive action that can be taken in one step (buying|edificating|mortaging|selling|bargaining for example)

\
In other matter, as i'm not very familiarized with gym library, there are a couple of warnings that'd showed up, this lets you play, but I hope to understand soon. Also obvservatio_space it's only there so the environment can be published, because currently my observation_space it's not hommogeneus to be in a numpy ndarray.
## Finally
If you want to try it the same way I've done, take a look at my main file, also you can test all the models I have in agents (currently none). The actual use of the players makes you to define all in the class, so it´s easy to import all the players, this means that for two player that use the same agent, you would need to use two different classes, this it's not attached to the enviornment so feel free to do it in your own way
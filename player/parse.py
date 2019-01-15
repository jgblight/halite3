import copy
import json
import _pickle as pickle
import os
import time
import os.path
from player import hlt
from player.hlt.networking import send_command
from player.state import GameState
from collections import defaultdict

ARBITRARY_ID = -1

def load_replay_file(file_name):
    with open(file_name, 'rb') as f:
        data = json.loads(f.read())
    return data

def get_winning_player(data):
    ranked = sorted(data['game_statistics']['player_statistics'], key=lambda x: x['rank'])
    winner_id = ranked[0]['player_id']
    return [p for p in data['players'] if p['player_id'] == winner_id][0]

def parse_winner(file_name):
    data = load_replay_file(file_name)
    winner = get_winning_player(data)
    return parse_replay_data(data, winner['name'].split(" ")[0])

def parse_replay_file(file_name, player_name):
    data = load_replay_file(file_name)
    return parse_replay_data(data, player_name)

def parse_replay_data(data, player_name):
    player = [p for p in data['players'] if p['name'].split(" ")[0] == player_name][0]
    player_id = int(player['player_id'])
    my_shipyard = hlt.Shipyard(player_id, ARBITRARY_ID,
                               hlt.Position(player['factory_location']['x'], player['factory_location']['y']))
    other_shipyards = [
        hlt.Shipyard(p['player_id'], ARBITRARY_ID, hlt.Position(p['factory_location']['x'], p['factory_location']['y']))
        for p in data['players'] if int(p['player_id']) != player_id]
    width = data['production_map']['width']
    height = data['production_map']['height']
    max_turns = data['GAME_CONSTANTS']['MAX_TURNS']
    first_cells = []
    for y in range(len(data['production_map']['grid'])):
        row = []
        for x in range(len(data['production_map']['grid'][y])):
            row += [data['production_map']['grid'][y][x]['energy']]
        first_cells.append(row)
    frames = []
    for f in data['full_frames']:
        prev_cells = first_cells if len(frames) == 0 else frames[-1]
        new_cells = json.loads(json.dumps(prev_cells))
        for c in f['cells']:
            new_cells[c['y']][c['x']] = c['production']
        frames.append(new_cells)
    moves = [{} if str(player_id) not in f['moves'] else {m['id']: m['direction'] for m in f['moves'][str(player_id)] if
                                                          m['type'] == "m"} for f in data['full_frames']]
    ships = [{} if str(player_id) not in f['entities'] else {
        int(sid): hlt.Ship(player_id, int(sid), hlt.Position(ship['x'], ship['y']), ship['energy']) for sid, ship in
        f['entities'][str(player_id)].items()} for f in data['full_frames']]
    other_ships = [
        {int(sid): hlt.Ship(int(pid), int(sid), hlt.Position(ship['x'], ship['y']), ship['energy']) for pid, p in
         f['entities'].items() if
         int(pid) != player_id for sid, ship in p.items()} for f in data['full_frames']]
    first_my_dropoffs = [my_shipyard]
    first_them_dropoffs = other_shipyards
    my_dropoffs = []
    them_dropoffs = []
    spawns = []
    deposits = []
    energy = []
    for f in data['full_frames']:
        energy.append(f['energy'].get(str(player_id)))
        deposits.append(f['deposited'].get(str(player_id)))
        new_my_dropoffs = copy.deepcopy(first_my_dropoffs if len(my_dropoffs) == 0 else my_dropoffs[-1])
        new_them_dropoffs = copy.deepcopy(first_them_dropoffs if len(them_dropoffs) == 0 else them_dropoffs[-1])
        spawn = False
        for e in f['events']:
            if e['type'] == 'spawn' and int(e['owner_id']) == player_id:
                spawn = True
            if e['type'] == 'construct':
                if int(e['owner_id']) == player_id:
                    new_my_dropoffs.append(
                        hlt.Dropoff(player_id, ARBITRARY_ID, hlt.Position(e['location']['x'], e['location']['y'])))
                else:
                    new_them_dropoffs.append(
                        hlt.Dropoff(e['owner_id'], ARBITRARY_ID, hlt.Position(e['location']['x'], e['location']['y'])))
        my_dropoffs.append(new_my_dropoffs)
        them_dropoffs.append(new_them_dropoffs)
        spawns.append(spawn)

    turns = [ float(x)/max_turns for x in range(len(data['full_frames']))]
    return [ GameState(*args) for args in zip(turns, frames, moves, ships, other_ships, my_dropoffs, them_dropoffs, spawns, deposits, energy) ]


def parse_replay_folder(folder_name, max_files=None):
    replay_buffer = []
    for file_name in sorted(os.listdir(folder_name)):
        if max_files is not None and len(replay_buffer) >= max_files:
            break
        else:
            replay_buffer.append(parse_winner(os.path.join(folder_name, file_name)))
    return replay_buffer

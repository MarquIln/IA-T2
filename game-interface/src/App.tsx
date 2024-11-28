import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [trainingStatus, setTrainingStatus] = useState('Not Started');

  interface GameState {
    message: string;
    board: string[];
  }

  const [gameState, setGameState] = useState<GameState | null>(null);
  const [currentPlayer, setCurrentPlayer] = useState<string | null>('X');
  const [board, setBoard] = useState(Array(9).fill(null));

  const startTraining = async () => {
    try {
      setTrainingStatus('In Progress...');
      await axios.post('http://localhost:5000/train');
      setTrainingStatus('Completed');
    } catch (error) {
      console.error('Error starting training:', error);
      setTrainingStatus('Error');
    }
  };

  const makeMove = async (index: number) => {
    if (board[index] || trainingStatus !== 'Completed' || currentPlayer !== 'X') return;

    const newBoard = [...board];
    newBoard[index] = 'X';
    setBoard(newBoard);

    try {
      const response = await axios.post('http://localhost:5000/simulate', { board: newBoard });
      const { board: updatedBoard, message } = response.data;

      setBoard(updatedBoard);
      setGameState({ message, board: updatedBoard });

      if (!message.includes('continua')) {
        setCurrentPlayer(null);
      } else {
        setCurrentPlayer('X');
      }
    } catch (error) {
      console.error('Error simulating move:', error);
    }
  };

  const renderSquare = (index: number) => {
    return (
      <button
        className="square btn"
        onClick={() => makeMove(index)}
        style={{
          cursor: trainingStatus === 'Completed' ? 'pointer' : 'not-allowed',
        }}
        disabled={!!board[index] || trainingStatus !== 'Completed'}
      >
        {board[index]}
      </button>
    );
  };

  return (
    <div className="container mt-5 d-flex justify-content-center align-items-center">
      <div className="text-center">
        <h1 className="mb-4 main-heading">Neural Network Training and Game</h1>
        <div className="mb-4">
          <h2 className="mb-3 training-heading">Train Neural Network</h2>
          <button className="btn btn-start" onClick={startTraining}>
            Start Training
          </button>
          <p className="mt-2 status-text">Status: {trainingStatus}</p>
        </div>
        <div>
          <h2 className="mb-3 game-heading">Tic-Tac-Toe Game</h2>
          <div style={{ display: 'inline-block' }}>
            <div
              className="game-board"
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '5px',
                marginBottom: '15px',
              }}
            >
              {Array.from({ length: 9 }).map((_, index) => renderSquare(index))}
            </div>
            {gameState && <p className="game-message">{gameState.message}</p>}
          </div>
          {trainingStatus !== 'Completed' && (
            <p className="text-danger">Please train the neural network before playing.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;

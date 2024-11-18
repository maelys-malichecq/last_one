import pytest
import os
import time
from pybacktestchain.blockchain import Block, Blockchain, load_blockchain, remove_blockchain
import hashlib

@pytest.fixture
def test_blockchain():
    """Fixture to create and clean up a test blockchain."""
    blockchain_name = "backtest"
    blockchain = Blockchain(name=blockchain_name)
    yield blockchain

def test_block_initialization():
    """Test Block initialization and hash calculation."""
    block = Block(name_backtest="Test Block", data="Test Data", previous_hash="0")
    assert block.name_backtest == "Test Block"
    assert block.data == "Test Data"
    assert block.previous_hash == "0"
    assert len(block.hash) == 64  # SHA-256 hash length


def test_block_hash_property():
    """Ensure the hash is calculated correctly."""
    block = Block(name_backtest="Block Test", data="Some Data", previous_hash="123")
    calculated_hash = hashlib.sha256(
        (str(block.timestamp) + block.name_backtest + block.data + block.previous_hash).encode()
    ).hexdigest()
    assert block.hash == calculated_hash


def test_blockchain_initialization(test_blockchain):
    """Test Blockchain initialization with a genesis block."""
    assert test_blockchain.name == "backtest"
    assert len(test_blockchain.chain) == 1  # Genesis block
    genesis_block = test_blockchain.chain[0]
    assert genesis_block.name_backtest == "Genesis Block"
    assert genesis_block.previous_hash == "0"


def test_adding_blocks(test_blockchain):
    """Test adding blocks to the blockchain."""
    test_blockchain.add_block(name="Block 1", data="Data 1")
    test_blockchain.add_block(name="Block 2", data="Data 2")

    assert len(test_blockchain.chain) == 3  # Genesis + 2 new blocks
    block_1 = test_blockchain.chain[1]
    block_2 = test_blockchain.chain[2]

    assert block_1.name_backtest == "Block 1"
    assert block_1.data == "Data 1"
    assert block_2.name_backtest == "Block 2"
    assert block_2.data == "Data 2"
    assert block_2.previous_hash == block_1.hash


def test_blockchain_validation(test_blockchain):
    """Test blockchain validation."""
    test_blockchain.add_block(name="Block 1", data="Data 1")
    test_blockchain.add_block(name="Block 2", data="Data 2")
    assert test_blockchain.is_valid()

    # Tamper with the chain
    test_blockchain.chain[1].data = "Tampered Data"
    assert not test_blockchain.is_valid()


def test_blockchain_persistence(test_blockchain):
    """Test storing and loading the blockchain."""
    test_blockchain.add_block(name="Block 1", data="Data 1")
    test_blockchain.add_block(name="Block 2", data="Data 2")

    # Ensure blockchain is persisted
    loaded_blockchain = load_blockchain(test_blockchain.name)
    assert len(loaded_blockchain.chain) == 3
    assert loaded_blockchain.chain[1].name_backtest == "Block 1"
    assert loaded_blockchain.chain[2].name_backtest == "Block 2"

    # Cleanup
    remove_blockchain(test_blockchain.name)
    assert not os.path.exists(f"blockchain/{test_blockchain.name}.pkl")

def test_blockchain_removal(test_blockchain):
    """Test removing the blockchain file."""
    assert os.path.exists(f"blockchain/{test_blockchain.name}.pkl")
    test_blockchain.remove_blockchain()
    assert not os.path.exists(f"blockchain/{test_blockchain.name}.pkl")

def test_blockchain_to_string(test_blockchain):
    """Test the __str__ method of the blockchain."""
    test_blockchain.add_block(name="Block 1", data="Data 1")
    blockchain_str = str(test_blockchain)
    assert "Block 1" in blockchain_str
    assert "Genesis Block" in blockchain_str
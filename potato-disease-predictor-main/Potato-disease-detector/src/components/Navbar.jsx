import React from 'react'
import { FaLeaf } from "react-icons/fa";
import "../index.css"
const Navbar = () => {
  return (
    <nav className="navbar">
      <h1>Plant Leaf Disease Detector App</h1>
      <div className="leaf-icon"><FaLeaf size={45} /></div>
    </nav>
  );
}

export default Navbar
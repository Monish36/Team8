<?php

$users = [
    'admin' => 'password123'
];

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $username = $_POST['username'];
    $password = $_POST['password'];

    if (isset($users[$username]) && $users[$username] == $password) {
        header('Location: dashboard.html');
        exit();
    } else {
        echo "Invalid credentials!";
    }
}
?>

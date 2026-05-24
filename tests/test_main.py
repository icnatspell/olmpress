from olmpress import main


def test_main_prints_greeting(capsys):
    main()
    assert "olmpress" in capsys.readouterr().out.lower()
